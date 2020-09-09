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
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc


def _sort_columns(feature_columns):
    return sorted(feature_columns, key=lambda col: col.name)


def _validate_numeric_column(feature_column):
    if len(feature_column.shape) > 1:
        return "Matrix numeric features are not allowed, " "found feature {} with shape {}".format(
            feature_column.key, feature_column.shape
        )
    elif feature_column.shape[0] != 1:
        return "Vector numeric features are not allowed, " "found feature {} with shape {}".format(
            feature_column.key, feature_column.shape
        )


def _validate_categorical_column(feature_column):
    if not isinstance(feature_column, fc.IdentityCategoricalColumn):
        return (
            "Only acceptable categorical columns for feeding "
            "embeddings are identity, found column {} of type {}. "
            "Consider using NVTabular online preprocessing to perform "
            "categorical transformations".format(feature_column.name, type(feature_column).__name__)
        )


def _validate_dense_feature_columns(feature_columns):
    _errors = []
    for feature_column in feature_columns:
        if isinstance(feature_column, fc.CategoricalColumn):
            if not isinstance(feature_column, fc.BucketizedColumn):
                _errors.append(
                    "All feature columns must be dense, found categorical "
                    "column {} of type {}. Please wrap categorical columns "
                    "in embedding or indicator columns before passing".format(
                        feature_column.name, type(feature_column).__name__
                    )
                )
            else:
                _errors.append(
                    "Found bucketized column {}. ScalarDenseFeatures layer "
                    "cannot apply bucketization preprocessing. Consider using "
                    "NVTabular to do preprocessing offline".format(feature_column.name)
                )
        elif isinstance(feature_column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            _errors.append(_validate_categorical_column(feature_column.categorical_column))

        elif isinstance(feature_column, fc.NumericColumn):
            _errors.append(_validate_numeric_column(feature_column))

    _errors = list(filter(lambda e: e is not None, _errors))
    if len(_errors) > 0:
        msg = "Found issues with columns passed to ScalarDenseFeatures:"
        msg += "\n\t".join(_errors)
        raise ValueError(_errors)


def _validate_stack_dimensions(feature_columns):
    dims = []
    for feature_column in feature_columns:
        if isinstance(feature_column, fc.EmbeddingColumn):
            dimension = feature_column.dimension
        elif isinstance(feature_column, fc.IndicatorColumn):
            dimension = feature_column.categorical_column.num_buckets
        else:
            dimension = feature_column.shape[0]

        dims.append(dimension)

    dim0 = dims[0]
    if not all(dim == dim0 for dim in dims[1:]):
        dims = ", ".join(map(str, dims))
        raise ValueError(
            "'stack' aggregation requires all categorical "
            "embeddings and continuous features to have same "
            "size. Found dimensions {}".format(dims)
        )


class ScalarDenseFeatures(tf.keras.layers.Layer):
    """
    Layer which maps one-hot categorical and scalar numeric features to
    a dense embedding. Meant to reproduce the API exposed by
    `tf.keras.layers.DenseFeatures` while reducing overhead for the
    case of one-hot categorical and scalar numeric features.

    Uses TensorFlow `feature_column`s to represent inputs to the layer, but
    does not perform any preprocessing associated with those columns. As such,
    it should only be passed `numeric_column`s and their subclasses,
    `embedding_column` and `indicator_column`. Preprocessing functionality should
    be moved to NVTabular.

    Note that caategorical columns should be wrapped in embedding or
    indicator columns first, consistent with the API used by
    `tf.keras.layers.DenseFeatures`.

    Example usage::

        column_a = tf.feature_column.numeric_column("a", (1,))
        column_b = tf.feature_column.categorical_column_with_identity("b", 100)
        column_b_embedding = tf.feature_column.embedding_column(column_b, 4)

        inputs = {
            "a": tf.keras.Input(name="a", shape=(1,), dtype=tf.float32),
            "b": tf.keras.Input(name="b", shape=(1,), dtype=tf.int64)
        }
        x = ScalarDenseFeatures([column_a, column_b_embedding])(inputs)

    Parameters
    ----------
    feature_columns : list of `tf.feature_column`
        feature columns describing the inputs to the layer
    """

    def __init__(self, feature_columns, aggregation="concat", name=None, **kwargs):
        # sort feature columns to make layer independent of column order
        feature_columns = _sort_columns(feature_columns)
        _validate_dense_feature_columns(feature_columns)

        assert aggregation in ("concat", "stack")
        if aggregation == "stack":
            _validate_stack_dimensions(feature_columns)

        self.feature_columns = feature_columns
        self.aggregation = aggregation
        super(ScalarDenseFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
        assert all(shape[1] == 1 for shape in input_shapes.values())

        self.embedding_tables = {}
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                continue

            feature_name = feature_column.categorical_column.key
            num_buckets = feature_column.categorical_column.num_buckets
            if isinstance(feature_column, fc.EmbeddingColumn):
                self.embedding_tables[feature_name] = self.add_weight(
                    name="{}/embedding_weights".format(feature_name),
                    trainable=True,
                    initializer="glorot_normal",
                    shape=(num_buckets, feature_column.dimension),
                )
            else:
                self.embedding_tables[feature_name] = self.add_weight(
                    name="{}/embedding_weights".format(feature_name),
                    trainable=False,
                    initializer=tf.constant_initializer(np.eye(num_buckets)),
                    shape=(num_buckets, num_buckets),
                )
        self.built = True

    def call(self, inputs):
        features = []
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                features.append(inputs[feature_column.name])
            else:
                feature_name = feature_column.categorical_column.name
                table = self.embedding_tables[feature_name]
                embeddings = tf.gather(table, inputs[feature_name][:, 0])
                features.append(embeddings)

        if self.aggregation == "stack":
            return tf.stack(features, axis=1)
        return tf.concat(features, axis=1)

    def compute_output_shape(self, input_shapes):
        input_shape = [i for i in input_shapes.values()][0]
        if self.aggregation == "concat":
            output_dim = len(self.numeric_features) + sum(
                [shape[-1] for shape in self.embedding_shapes.values()]
            )
            return (input_shape[0], output_dim)
        else:
            embedding_dim = [i for i in self.embedding_shapes.values()][0]
            return (input_shape[0], len(self.embedding_shapes), embedding_dim)

    def get_config(self):
        return {
            "feature_columns": self.feature_columns,
            "aggregation": self.aggregation,
        }


def _validate_linear_feature_columns(feature_columns):
    _errors = []
    for feature_column in feature_columns:
        if isinstance(feature_column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            _errors.append(
                "Only pass categorical or numeric columns to ScalarLinearFeatures "
                "layer, found column {} of type".format(feature_column)
            )
        elif isinstance(feature_column, fc.NumericColumn):
            _errors.append(_validate_numeric_column(feature_column))
        else:
            _errors.append(_validate_categorical_column(feature_column))

    _errors = list(filter(lambda e: e is not None, _errors))
    if len(_errors) > 0:
        msg = "Found issues with columns passed to ScalarDenseFeatures:"
        msg += "\n\t".join(_errors)
        raise ValueError(_errors)


# TODO: is there a clean way to combine these two layers
# into one, maybe with a "sum" aggregation? Major differences
# seem to be whether categorical columns are wrapped in
# embeddings and the numeric matmul, both of which seem
# reasonably easy to check. At the very least, we should
# be able to subclass I think?
class ScalarLinearFeatures(tf.keras.layers.Layer):
    """
    Layer which implements a linear combination of one-hot categorical
    and scalar numeric features. Based on the "wide" branch of the Wide & Deep
    network architecture.

    Uses TensorFlow `feature_column`s to represent inputs to the layer, but
    does not perform any preprocessing associated with those columns. As such,
    it should only be passed `numeric_column` and
    `categorical_column_with_identity`. Preprocessing functionality should
    be moved to NVTabular.

    Also note that, unlike ScalarDenseFeatures, categorical columns should
    NOT be wrapped in embedding or indicator columns first.

    Example usage::

        column_a = tf.feature_column.numeric_column("a", (1,))
        column_b = tf.feature_column.categorical_column_with_identity("b", 100)

        inputs = {
            "a": tf.keras.Input(name="a", shape=(1,), dtype=tf.float32),
            "b": tf.keras.Input(name="b", shape=(1,), dtype=tf.int64)
        }
        x = ScalarLinearFeatures([column_a, column_b])(inputs)

    Parameters
    ----------
    feature_columns : list of tf.feature_column
        feature columns describing the inputs to the layer
    """

    def __init__(self, feature_columns, name=None, **kwargs):
        feature_columns = _sort_columns(feature_columns)
        _validate_linear_feature_columns(feature_columns)

        self.feature_columns = feature_columns
        super(ScalarLinearFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
        assert all(shape[1] == 1 for shape in input_shapes.values())

        # TODO: I've tried combining all the categorical tables
        # into a single giant lookup op, but it ends up turning
        # out the adding the offsets to lookup indices at call
        # time ends up being much slower due to kernel overhead
        # Still, a better (and probably custom) solutions would
        # probably be desirable
        numeric_kernel_dim = 0
        self.embedding_tables = {}
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                numeric_kernel_dim += feature_column.shape[0]
                continue

            self.embedding_tables[feature_column.key] = self.add_weight(
                name="{}/embedding_weights".format(feature_column.key),
                initializer="zeros",
                trainable=True,
                shape=(feature_column.num_buckets, 1),
            )
        if numeric_kernel_dim > 0:
            self.embedding_tables["numeric"] = self.add_weight(
                name="numeric/embedding_weights",
                initializer="zeros",
                trainable=True,
                shape=(numeric_kernel_dim, 1),
            )

        self.bias = self.add_weight(name="bias", initializer="zeros", trainable=True, shape=(1,))
        self.built = True

    def call(self, inputs):
        x = self.bias
        numeric_inputs = []
        for feature_column in self.feature_columns:
            if isinstance(feature_column, fc.NumericColumn):
                numeric_inputs.append(inputs[feature_column.key])
            else:
                table = self.embedding_tables[feature_column.key]
                x = x + tf.gather(table, inputs[feature_column.key][:, 0])

        if len(numeric_inputs) > 0:
            numerics = tf.concat(numeric_inputs, axis=1)
            x = x + tf.matmul(numerics, self.embedding_tables["numeric"])
        return x

    def compute_output_shape(self, input_shapes):
        batch_size = [i for i in input_shapes.values()][0].shape[0]
        return (batch_size, 1)

    def get_config(self):
        return {
            "feature_columns": self.feature_columns,
        }
