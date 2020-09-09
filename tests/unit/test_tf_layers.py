#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
        tf.feature_column.numeric_column("a", (1,)),
        tf.feature_column.categorical_column_with_identity("b", 100),
        tf.feature_column.categorical_column_with_identity("c", 5),
    )


def get_bad_feature_columns():
    return (
        tf.feature_column.numeric_column("a", (5,)),
        tf.feature_column.categorical_column_with_hash_bucket("b", 100),
    )


@pytest.mark.parametrize("aggregation", ["stack", "concat"])
def test_dense_embedding_layer(aggregation):
    col_a, col_b, col_c = get_good_feature_columns()
    col_b_embedding = tf.feature_column.embedding_column(col_b, 8)
    col_c_embedding = tf.feature_column.indicator_column(col_c)

    # should raise ValueError if passed categorical columns
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarDenseFeatures(
            [col_a, col_b, col_c], aggregation=aggregation
        )

    if aggregation == "stack":
        # can't pass numeric to stack aggregation unless dims are 1
        with pytest.raises(ValueError):
            embedding_layer = layers.ScalarDenseFeatures(
                [col_a, col_b_embedding, col_c_embedding], aggregation=aggregation
            )
        # can't have mismatched dims with stack aggregation
        with pytest.raises(ValueError):
            embedding_layer = layers.ScalarDenseFeatures(
                [col_b_embedding, col_c_embedding], aggregation=aggregation
            )

        # reset b embedding to have matching dims
        col_b_embedding = tf.feature_column.embedding_column(col_b, 5)
        cols = [col_b_embedding, col_c_embedding]
    else:
        cols = [col_a, col_b_embedding, col_c_embedding]

    embedding_layer = layers.ScalarDenseFeatures(cols, aggregation=aggregation)

    inputs = {
        "a": tf.keras.Input(name="a", shape=(1,), dtype=tf.float32),
        "b": tf.keras.Input(name="b", shape=(1,), dtype=tf.int64),
        "c": tf.keras.Input(name="c", shape=(1,), dtype=tf.int64),
    }
    if aggregation == "stack":
        inputs.pop("a")

    output = embedding_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile("sgd", "mse")

    # TODO: check for out-of-range categorical behavior
    a = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    b = np.array([44, 21, 32])
    c = np.array([0, 4, 2])
    x = {"a": a[:, None], "b": b[:, None], "c": c[:, None]}
    if aggregation == "stack":
        x.pop("a")

    b_embedding_table = embedding_layer.embedding_tables["b"].numpy()
    b_embedding_rows = b_embedding_table[b]

    # check that shape and values match up
    y_hat = model.predict(x)
    assert y_hat.shape[0] == 3
    if aggregation == "stack":
        assert len(y_hat.shape) == 3
        assert y_hat.shape[1] == len(x)
        assert y_hat.shape[2] == 5

        assert (y_hat[:, 0] == b_embedding_rows).all()
        y_c = y_hat[:, 1]

    elif aggregation == "concat":
        assert len(y_hat.shape) == 2
        assert y_hat.shape[1] == 1 + 8 + 5

        assert (y_hat[:, 0] == a).all()
        assert (y_hat[:, 1:9] == b_embedding_rows).all()

        y_c = y_hat[:, 9:]

    nonzero_c_idx = np.where(y_c != 0)
    assert (nonzero_c_idx[1] == c).all()
    assert (y_c[nonzero_c_idx] == 1).all()

    # make sure unusable columns get flagged
    bad_col_a, bad_col_b = get_bad_feature_columns()
    bad_col_b_embedding = tf.feature_column.embedding_column(
        bad_col_b, col_b_embedding.dimension
    )
    with pytest.raises(ValueError):
        # vector numeric should raise, even though dims match
        embedding_layer = layers.ScalarDenseFeatures(
            [bad_col_a, col_b_embedding, col_c_embedding], aggregation=aggregation
        )
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarDenseFeatures(
            [bad_col_b_embedding, col_c_embedding], aggregation=aggregation
        )


def test_linear_embedding_layer():
    col_a, col_b, col_c = get_good_feature_columns()
    col_b_embedding = tf.feature_column.embedding_column(col_b, 8)

    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarLinearFeatures([col_a, col_b_embedding, col_c])
    embedding_layer = layers.ScalarLinearFeatures([col_a, col_b, col_c])

    inputs = {
        "a": tf.keras.Input(name="a", shape=(1,), dtype=tf.float32),
        "b": tf.keras.Input(name="b", shape=(1,), dtype=tf.int64),
        "c": tf.keras.Input(name="c", shape=(1,), dtype=tf.int64),
    }

    output = embedding_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile("sgd", "mse")

    # TODO: check for out-of-range categorical behavior
    a = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    b = np.array([44, 21, 32])
    c = np.array([0, 4, 2])
    x = {"a": a[:, None], "b": b[:, None], "c": c[:, None]}

    y_hat = model.predict(x)
    assert (y_hat == 0).all()

    # do one iteration of training to make sure values
    # match up, since all initialized to zero
    y = np.array([-0.5, 1.2, 3.4])[:, None]
    model.fit(x, y, batch_size=3, epochs=1, verbose=0)
    y_hat = model.predict(x)[:, 0]

    a_weight = embedding_layer.embedding_tables["numeric"].numpy()[0, 0]
    b_weights = embedding_layer.embedding_tables["b"].numpy()[b][:, 0]
    c_weights = embedding_layer.embedding_tables["c"].numpy()[c][:, 0]
    bias = embedding_layer.bias.numpy()[0]

    rtol = 1e-6
    expected_y_hat = a_weight * a + b_weights + c_weights + bias
    assert np.isclose(y_hat, expected_y_hat, rtol=rtol).all()

    # make sure unusable columns get flagged
    bad_col_a, bad_col_b = get_bad_feature_columns()
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarLinearFeatures([bad_col_a, col_b, col_c])
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarLinearFeatures([col_a, bad_col_b, col_c])


@pytest.mark.parametrize("embedding_dim", [1, 4, 16])
@pytest.mark.parametrize("num_features", [1, 16, 64])
@pytest.mark.parametrize(
    "interaction_type", [None, "field_all", "field_each", "field_interaction"]
)
@pytest.mark.parametrize("self_interaction", [True, False])
def test_dot_product_interaction_layer(
    embedding_dim, num_features, interaction_type, self_interaction
):
    if num_features == 1 and not self_interaction:
        return

    input = tf.keras.Input(
        name="x", shape=(num_features, embedding_dim), dtype=tf.float32
    )
    interaction_layer = layers.DotProductInteraction(interaction_type, self_interaction)
    output = interaction_layer(input)
    model = tf.keras.Model(inputs=input, outputs=output)
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

    rtol = 1e-5
    atol = 1e-6
    frac_correct = 1.0
    match = np.isclose(expected_output, y_hat, rtol=rtol, atol=atol)
    assert match.mean() >= frac_correct
