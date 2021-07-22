#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
layers = pytest.importorskip("nvtabular.framework_utils.tensorflow.layers")


def get_good_feature_columns():
    return (
        tf.feature_column.numeric_column("scalar_continuous", (1,)),
        tf.feature_column.numeric_column("vector_continuous", (128,)),
        tf.feature_column.categorical_column_with_identity("one_hot", 100),
        tf.feature_column.categorical_column_with_identity("multi_hot", 5),
    )


def get_bad_feature_columns():
    return (
        tf.feature_column.numeric_column("a", (5, 10)),
        tf.feature_column.categorical_column_with_hash_bucket("b", 100),
    )


def _compute_expected_multi_hot(table, values, nnzs, combiner):
    rows = []
    start_idx = 0
    for nnz in nnzs:
        vals = values[start_idx : start_idx + nnz]
        vectors = table[vals]
        if combiner == "sum":
            rows.append(vectors.sum(axis=0))
        else:
            rows.append(vectors.mean(axis=0))
        start_idx += nnz
    return np.array(rows)


@pytest.mark.parametrize("aggregation", ["stack", "concat"])
@pytest.mark.parametrize("combiner", ["sum", "mean"])  # TODO: add sqrtn
def test_dense_embedding_layer(aggregation, combiner):
    raw_good_columns = get_good_feature_columns()
    scalar_numeric, vector_numeric, one_hot, multi_hot = raw_good_columns
    one_hot_embedding = tf.feature_column.indicator_column(one_hot)
    multi_hot_embedding = tf.feature_column.embedding_column(multi_hot, 8, combiner=combiner)

    # should raise ValueError if passed categorical columns
    with pytest.raises(ValueError):
        embedding_layer = layers.DenseFeatures(raw_good_columns, aggregation=aggregation)

    if aggregation == "stack":
        # can't pass numeric to stack aggregation unless dims are 1
        with pytest.raises(ValueError):
            embedding_layer = layers.DenseFeatures(
                [
                    scalar_numeric,
                    vector_numeric,
                    one_hot_embedding,
                    multi_hot_embedding,
                ],
                aggregation=aggregation,
            )
        # can't have mismatched dims with stack aggregation
        with pytest.raises(ValueError):
            embedding_layer = layers.DenseFeatures(
                [one_hot_embedding, multi_hot_embedding], aggregation=aggregation
            )

        # reset b embedding to have matching dims
        multi_hot_embedding = tf.feature_column.embedding_column(multi_hot, 100, combiner=combiner)
        cols = [one_hot_embedding, multi_hot_embedding]
    else:
        cols = [scalar_numeric, vector_numeric, one_hot_embedding, multi_hot_embedding]

    embedding_layer = layers.DenseFeatures(cols, aggregation=aggregation)
    inputs = {
        "scalar_continuous": tf.keras.Input(name="scalar_continuous", shape=(1,), dtype=tf.float32),
        "vector_continuous": tf.keras.Input(
            name="vector_continuous__values", shape=(1,), dtype=tf.float32
        ),
        "one_hot": tf.keras.Input(name="one_hot", shape=(1,), dtype=tf.int64),
        "multi_hot": (
            tf.keras.Input(name="multi_hot__values", shape=(1,), dtype=tf.int64),
            tf.keras.Input(name="multi_hot__nnzs", shape=(1,), dtype=tf.int64),
        ),
    }
    if aggregation == "stack":
        inputs.pop("scalar_continuous")
        inputs.pop("vector_continuous")

    output = embedding_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile("sgd", "mse")

    # TODO: check for out-of-range categorical behavior
    scalar = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    vector = np.random.randn(3, 128).astype("float32")
    one_hot = np.array([44, 21, 32])
    multi_hot_values = np.array([0, 2, 1, 4, 1, 3, 1])
    multi_hot_nnzs = np.array([1, 2, 4])
    x = {
        "scalar_continuous": scalar[:, None],
        "vector_continuous": vector.flatten()[:, None],
        "one_hot": one_hot[:, None],
        "multi_hot": (multi_hot_values[:, None], multi_hot_nnzs[:, None]),
    }
    if aggregation == "stack":
        x.pop("scalar_continuous")
        x.pop("vector_continuous")

    multi_hot_embedding_table = embedding_layer.embedding_tables["multi_hot"].numpy()
    multi_hot_embedding_rows = _compute_expected_multi_hot(
        multi_hot_embedding_table, multi_hot_values, multi_hot_nnzs, combiner
    )

    # check that shape and values match up
    y_hat = model(x).numpy()
    assert y_hat.shape[0] == 3
    if aggregation == "stack":
        assert len(y_hat.shape) == 3
        # len of columns is 2 because of mh (vals, nnzs) struct
        assert y_hat.shape[1] == (len(x))
        assert y_hat.shape[2] == 100
        np.testing.assert_allclose(y_hat[:, 0], multi_hot_embedding_rows, rtol=1e-05)
        y_c = y_hat[:, 1]

    elif aggregation == "concat":
        assert len(y_hat.shape) == 2
        assert y_hat.shape[1] == 1 + 100 + 8 + 128

        assert (y_hat[:, 108] == scalar).all()
        assert (y_hat[:, 109:] == vector).all()
        np.testing.assert_allclose(y_hat[:, :8], multi_hot_embedding_rows, rtol=1e-05)

        y_c = y_hat[:, 8:108]

    nonzero_c_idx = np.where(y_c != 0)
    assert (nonzero_c_idx[1] == one_hot).all()
    assert (y_c[nonzero_c_idx] == 1).all()

    # make sure unusable columns get flagged
    bad_col_a, bad_col_b = get_bad_feature_columns()
    bad_col_b_embedding = tf.feature_column.indicator_column(bad_col_b)
    with pytest.raises(ValueError):
        # matrix numeric should raise, even though dims match
        embedding_layer = layers.DenseFeatures(
            [bad_col_a, one_hot_embedding, multi_hot_embedding], aggregation=aggregation
        )
    with pytest.raises(ValueError):
        embedding_layer = layers.DenseFeatures(
            [bad_col_b_embedding, multi_hot_embedding], aggregation=aggregation
        )


def test_linear_embedding_layer():
    raw_good_columns = get_good_feature_columns()
    scalar_numeric, vector_numeric, one_hot_col, multi_hot_col = raw_good_columns
    one_hot_embedding = tf.feature_column.indicator_column(one_hot_col)

    with pytest.raises(ValueError):
        embedding_layer = layers.LinearFeatures([scalar_numeric, one_hot_embedding])
    embedding_layer = layers.LinearFeatures(raw_good_columns)

    inputs = {
        "scalar_continuous": tf.keras.Input(name="scalar_continuous", shape=(1,), dtype=tf.float32),
        "vector_continuous": tf.keras.Input(
            name="vector_continuous__values", shape=(1,), dtype=tf.float32
        ),
        "one_hot": tf.keras.Input(name="one_hot", shape=(1,), dtype=tf.int64),
        "multi_hot": (
            tf.keras.Input(name="multi_hot__values", shape=(1,), dtype=tf.int64),
            tf.keras.Input(name="multi_hot__nnzs", shape=(1,), dtype=tf.int64),
        ),
    }

    output = embedding_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile("sgd", "mse")

    # TODO: check for out-of-range categorical behavior
    scalar = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    vector = np.random.randn(3, 128).astype("float32")
    one_hot = np.array([44, 21, 32])
    multi_hot_values = np.array([0, 2, 1, 4, 1, 3, 1])
    multi_hot_nnzs = np.array([1, 2, 4])
    x = {
        "scalar_continuous": scalar[:, None],
        "vector_continuous": vector.flatten()[:, None],
        "one_hot": one_hot[:, None],
        "multi_hot": (multi_hot_values[:, None], multi_hot_nnzs[:, None]),
    }

    y_hat = model(x).numpy()
    assert (y_hat == 0).all()

    # do one iteration of training to make sure values
    # match up, since all initialized to zero
    y = np.array([-0.5, 1.2, 3.4])[:, None]
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = tf.reduce_mean((y_hat - y) ** 2)
        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
    y_hat = model(x).numpy()[:, 0]

    numeric_weight = embedding_layer.embedding_tables["numeric"].numpy()
    one_hot_weights = embedding_layer.embedding_tables["one_hot"].numpy()[one_hot][:, 0]
    multi_hot_weights = _compute_expected_multi_hot(
        embedding_layer.embedding_tables["multi_hot"].numpy(),
        multi_hot_values,
        multi_hot_nnzs,
        "sum",
    )[:, 0]
    bias = embedding_layer.bias.numpy()[0]

    rtol = 1e-5
    numeric = np.concatenate([scalar[:, None], vector], axis=1)
    expected_y_hat = (numeric @ numeric_weight)[:, 0] + one_hot_weights + multi_hot_weights + bias
    assert np.isclose(y_hat, expected_y_hat, rtol=rtol).all()

    # make sure unusable columns get flagged
    bad_col_a, bad_col_b = get_bad_feature_columns()
    with pytest.raises(ValueError):
        embedding_layer = layers.LinearFeatures([bad_col_a, one_hot_col, multi_hot_col])
    with pytest.raises(ValueError):
        embedding_layer = layers.LinearFeatures([scalar_numeric, bad_col_b, multi_hot_col])


@pytest.mark.parametrize("embedding_dim", [1, 4, 16])
@pytest.mark.parametrize("num_features", [1, 16, 64])
@pytest.mark.parametrize("interaction_type", [None, "field_all", "field_each", "field_interaction"])
@pytest.mark.parametrize("self_interaction", [True, False])
def test_dot_product_interaction_layer(
    embedding_dim, num_features, interaction_type, self_interaction
):
    if num_features == 1 and not self_interaction:
        return

    input_ = tf.keras.Input(name="x", shape=(num_features, embedding_dim), dtype=tf.float32)
    interaction_layer = layers.DotProductInteraction(interaction_type, self_interaction)
    output = interaction_layer(input_)
    model = tf.keras.Model(inputs=input_, outputs=output)
    model.compile("sgd", "mse")

    x = np.random.randn(8, num_features, embedding_dim).astype(np.float32)
    y_hat = model.predict(x)

    if self_interaction:
        expected_dim = num_features * (num_features + 1) // 2
    else:
        expected_dim = num_features * (num_features - 1) // 2
    assert y_hat.shape[1] == expected_dim

    if interaction_type is not None:
        W = interaction_layer.kernel.numpy()
    expected_outputs = []
    for i in range(num_features):
        j_start = i if self_interaction else i + 1
        for j in range(j_start, num_features):
            x_i = x[:, i]
            x_j = x[:, j]
            if interaction_type == "field_all":
                W_ij = W
            elif interaction_type == "field_each":
                W_ij = W[i].T
            elif interaction_type == "field_interaction":
                W_ij = W[i, j]

            if interaction_type is not None:
                x_i = x_i @ W_ij
            expected_outputs.append((x_i * x_j).sum(axis=1))
    expected_output = np.stack(expected_outputs).T

    rtol = 1e-1
    atol = 1e-2
    match = np.isclose(expected_output, y_hat, rtol=rtol, atol=atol)
    assert match.all()


def test_multihot_empty_rows():
    multi_hot = tf.feature_column.categorical_column_with_identity("multihot", 5)
    multi_hot_embedding = tf.feature_column.embedding_column(multi_hot, 8, combiner="sum")

    embedding_layer = layers.DenseFeatures([multi_hot_embedding])
    inputs = {
        "multihot": (
            tf.keras.Input(name="multihot__values", shape=(1,), dtype=tf.int64),
            tf.keras.Input(name="multihot__nnzs", shape=(1,), dtype=tf.int64),
        )
    }
    output = embedding_layer(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile("sgd", "binary_crossentropy")

    multi_hot_values = np.array([0, 2, 1, 4, 1, 3, 1])
    multi_hot_nnzs = np.array([1, 0, 2, 4, 0])
    x = {"multihot": (multi_hot_values[:, None], multi_hot_nnzs[:, None])}

    multi_hot_embedding_table = embedding_layer.embedding_tables["multihot"].numpy()
    multi_hot_embedding_rows = _compute_expected_multi_hot(
        multi_hot_embedding_table, multi_hot_values, multi_hot_nnzs, "sum"
    )

    y_hat = model(x).numpy()
    np.testing.assert_allclose(y_hat, multi_hot_embedding_rows, rtol=1e-06)
