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

import pytest

import numpy as np

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
        tf.feature_colum.categorical_column_with_hash_bucket("b", 100),
    )


@pytest.mark.parameterize("aggregation", ["stack", "concat"])
def test_dense_embedding_layer(aggregation):
    col_a, col_b, col_c = get_good_feature_columns()
    col_b_embedding = tf.feature_column.embedding_column(col_b, 8)
    col_c_embedding = tf.feature_column.indicator_column(col_c)

    # should raise ValueError if passed categorical columns
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarDenseFeatures([col_a, col_b, col_c], aggregation=aggregation)

    if aggregation == "stack":
        # can't pass numeric to stack aggregation
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

    embedding_layer = layers.ScalarDenseFeatures(
        [col_a, col_b_embedding, col_c_embedding], aggregation=aggregation
    )

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

    b_embedding_table = embedding_layer.embedding_tables["b"].numpy()
    b_embedding_rows = b_embedding_table[b]

    # check that shape and values match up
    y_hat = model.predict(x)
    assert y_hat.shape[0] == 3
    if aggregation == "stack":
        assert len(y_hat.shape) == 3
        assert y_hat.shape[1] == 3
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
    bad_col_b_embedding = tf.feature_column.embedding_column(bad_col_b, 8)
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarDenseFeatures(
            [bad_col_a, col_b_embedding, col_c_embedding], aggregation=aggregation
        )
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarDenseFeatures(
            [col_a, bad_col_b_embedding, col_c_embedding], aggregation=aggregation
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
    c_weights = embedding_layer.embedding_talbes["c"].numpy()[c][:, 0]
    bias = embedding_layer.bias.numpy()[0]

    assert y_hat == (a_weight * a + b_weights + c_weights + bias)

    # make sure unusable columns get flagged
    bad_col_a, bad_col_b = get_bad_feature_columns()
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarLinearFeatures([bad_col_a, col_b, col_c])
    with pytest.raises(ValueError):
        embedding_layer = layers.ScalarLinearFeatures([col_a, bad_col_b, col_c])
