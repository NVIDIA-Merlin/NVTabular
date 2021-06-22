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
import math
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2 as fc

# pylint has issues with TF array ops, so disable checks until fixed:
# https://github.com/PyCQA/pylint/issues/3613
# pylint: disable=no-value-for-parameter, unexpected-keyword-arg
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.tpu.tpu_embedding_v2_utils import (
    FeatureConfig,
    TableConfig,
)

from nvtabular.column_group import ColumnGroup
from nvtabular.framework_utils.tensorflow.features import TabularLayer, AsSparseLayer, ParseTokenizedText, \
    FilterFeatures
from nvtabular.ops import get_embedding_sizes
from nvtabular.tag import Tag
from nvtabular.workflow import Workflow


def _sort_columns(feature_columns):
    return sorted(feature_columns, key=lambda col: col.name)


def _validate_numeric_column(feature_column):
    if len(feature_column.shape) > 1:
        return "Matrix numeric features are not allowed, " "found feature {} with shape {}".format(
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
                    "Found bucketized column {}. DenseFeatures layer "
                    "cannot apply bucketization preprocessing. Consider using "
                    "NVTabular to do preprocessing offline".format(feature_column.name)
                )
        elif isinstance(feature_column, (fc.EmbeddingColumn, fc.IndicatorColumn)):
            _errors.append(_validate_categorical_column(feature_column.categorical_column))

        elif isinstance(feature_column, fc.NumericColumn):
            _errors.append(_validate_numeric_column(feature_column))

    _errors = list(filter(lambda e: e is not None, _errors))
    if len(_errors) > 0:
        msg = "Found issues with columns passed to DenseFeatures:"
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


def _categorical_embedding_lookup(table, inputs, feature_name, combiner):
    # check for sparse embeddings by name
    # build values and nnz tensors into ragged array, convert to sparse
    if isinstance(inputs[feature_name], tuple):
        values = inputs[feature_name][0][:, 0]
        row_lengths = inputs[feature_name][1][:, 0]
        x = tf.RaggedTensor.from_row_lengths(values, row_lengths).to_sparse()

        # use ragged array for sparse embedding lookup.
        # note we're using safe_embedding_lookup_sparse to handle empty rows
        # ( https://github.com/NVIDIA/NVTabular/issues/389 )
        embeddings = tf.nn.safe_embedding_lookup_sparse(table, x, None, combiner=combiner)
    else:
        embeddings = tf.gather(table, inputs[feature_name][:, 0])

    return embeddings


def _handle_continuous_feature(inputs, feature_column):
    if feature_column.shape[0] > 1:
        x = inputs[feature_column.name]
        if isinstance(x, tuple):
            x = x[0]
        return tf.reshape(x, (-1, feature_column.shape[0]))
    return inputs[feature_column.name]


class EmbeddingsLayer(TabularLayer):
    """Mimics the API of [TPUEmbedding-layer](https://github.com/tensorflow/recommenders/blob/main/tensorflow_recommenders/layers/embedding/tpu_embedding_layer.py#L221)
    from TF-recommenders, use this for efficient embeddings on CPU or GPU."""

    def __init__(self, feature_config: Dict[str, FeatureConfig], **kwargs):
        super().__init__(**kwargs)
        self.feature_config = feature_config
        self.convert_to_sparse = AsSparseLayer()
        self.filter_features = FilterFeatures(list(feature_config.keys()))

    def build(self, input_shapes):
        self.embedding_tables = {}
        tables: Dict[str, TableConfig] = {}
        for name, feature in self.feature_config.items():
            table: TableConfig = feature.table
            if table.name not in tables:
                tables[table.name] = table

        for name, table in tables.items():
            shape = (table.vocabulary_size, table.dim)
            self.embedding_tables[name] = self.add_weight(
                name="{}/embedding_weights".format(name),
                trainable=True,
                initializer=table.initializer,
                shape=shape,
            )
        super().build(input_shapes)

    def lookup_feature(self, name, val):
        table: TableConfig = self.feature_config[name].table
        table_var = self.embedding_tables[table.name]
        if isinstance(val, tf.SparseTensor):
            return tf.nn.safe_embedding_lookup_sparse(
                table_var, tf.cast(val, tf.int32), None, combiner=table.combiner
            )

        # embedded_outputs[name] = tf.gather(table_var, tf.cast(val, tf.int32))
        return tf.gather(table_var, tf.cast(val, tf.int32)[:, 0])

    def compute_output_shape(self, input_shapes):
        input_shapes = self.filter_features.compute_output_shape(input_shapes)
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        output_shapes = {}

        for name, val in input_shapes.items():
            output_shapes[name] = tf.TensorShape([batch_size, self.feature_config[name].table.dim])

        return super().compute_output_shape(output_shapes)

    def call(self, inputs, **kwargs):
        embedded_outputs = {}
        sparse_inputs = self.convert_to_sparse(self.filter_features(inputs))
        for name, val in sparse_inputs.items():
            embedded_outputs[name] = self.lookup_feature(name, val)

        return embedded_outputs

    @classmethod
    def from_nvt_workflow(cls, workflow: Workflow, combiner="mean") -> "EmbeddingsLayer":
        embedding_size = get_embedding_sizes(workflow)
        if isinstance(embedding_size, tuple):
            embedding_size = embedding_size[0]
        feature_config: Dict[str, FeatureConfig] = {}
        for name, (vocab_size, dim) in embedding_size.items():
            feature_config[name] = FeatureConfig(
                TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    name=name,
                    combiner=combiner,
                    initializer=init_ops_v2.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(dim)),
                )
            )

        return cls(feature_config)

    @classmethod
    def from_column_group(cls, column_group: ColumnGroup, embedding_dims=None, default_embedding_dim=64,
                          infer_embedding_sizes=True, combiner="mean", tags=None, tags_to_filter=None, **kwargs):
        if tags:
            column_group = column_group.get_tagged(tags, tags_to_filter=tags_to_filter)

        if infer_embedding_sizes:
            sizes = column_group.embedding_sizes()
        else:
            if not embedding_dims:
                embedding_dims = {}
            sizes = {}
            cardinalities = column_group.cardinalities()
            for key, cardinality in cardinalities.items():
                embedding_size = embedding_dims.get(key, default_embedding_dim)
                sizes[key] = (cardinality, embedding_size)

        feature_config: Dict[str, FeatureConfig] = {}
        for name, (vocab_size, dim) in sizes.items():
            feature_config[name] = FeatureConfig(
                TableConfig(
                    vocabulary_size=vocab_size,
                    dim=dim,
                    name=name,
                    combiner=combiner,
                    initializer=init_ops_v2.TruncatedNormal(mean=0.0, stddev=1 / math.sqrt(dim)),
                )
            )

        return cls(feature_config, **kwargs)


class TransformersTextEmbedding(TabularLayer):
    def __init__(self, transformer_model, max_text_length=None, output="pooler_output", trainable=False, **kwargs):
        super().__init__(trainable=trainable, **kwargs)
        self.parse_tokens = ParseTokenizedText(max_text_length=max_text_length)
        self.transformer_model = transformer_model
        self.transformer_output = output

    def call(self, inputs, **kwargs):
        tokenized = self.parse_tokens(inputs)
        outputs = {}
        for key, val in tokenized.items():
            if self.transformer_output == "pooler_output":
                outputs[key] = self.transformer_model(**val).pooler_output
            elif self.transformer_output == "last_hidden_state":
                outputs[key] = self.transformer_model(**val).last_hidden_state
            else:
                outputs[key] = self.transformer_model(**val)

        return outputs

    def compute_output_shape(self, input_shapes):
        batch_size = self.calculate_batch_size_from_input_shapes(input_shapes)

        # TODO: Handle all transformer output modes

        output_shapes, text_column_names = {}, []
        for name, val in input_shapes.items():
            if isinstance(val, tuple) and name.endswith(("/tokens", "/attention_mask")):
                text_column_names.append("/".join(name.split("/")[:-1]))

        for text_col in set(text_column_names):
            output_shapes[text_col] = tf.TensorShape([batch_size, self.transformer_model.config.hidden_size])

        return super().compute_output_shape(output_shapes)


class InputFeatures(TabularLayer):
    def __init__(self, continuous_layer=None, categorical_layer=None, text_embedding_layer=None, aggregation=None,
                 **kwargs):
        super(InputFeatures, self).__init__(aggregation=aggregation, **kwargs)
        self.categorical_layer = categorical_layer
        self.continuous_layer = continuous_layer
        self.text_embedding_layer = text_embedding_layer

        self.to_apply = []
        if continuous_layer:
            self.to_apply.append(continuous_layer)
        if categorical_layer:
            self.to_apply.append(categorical_layer)
        if text_embedding_layer:
            self.to_apply.append(text_embedding_layer)

        assert (self.to_apply is not []), "Please provide at least one input layer"

    def call(self, inputs, **kwargs):
        return self.to_apply[0](inputs, merge_with=self.to_apply[1:] if len(self.to_apply) > 1 else None)

    @classmethod
    def from_column_group(cls,
                          column_group: ColumnGroup,
                          continuous_tags=Tag.CONTINUOUS,
                          continuous_tags_to_filter=None,
                          categorical_tags=Tag.CATEGORICAL,
                          categorical_tags_to_filter=None,
                          text_model=None,
                          text_tags=Tag.TEXT_TOKENIZED,
                          text_tags_to_filter=None,
                          max_text_length=None,
                          aggregation=None,
                          **kwargs):
        continuous_layer, categorical_layer = None, None
        if continuous_tags:
            continuous_layer = TabularLayer.from_column_group(
                column_group,
                tags=continuous_tags,
                tags_to_filter=continuous_tags_to_filter)
        if categorical_tags:
            categorical_layer = EmbeddingsLayer.from_column_group(
                column_group,
                tags=categorical_tags,
                tags_to_filter=categorical_tags_to_filter)

        if text_model and not isinstance(text_model, TransformersTextEmbedding):
            text_model = TransformersTextEmbedding.from_column_group(
                column_group,
                tags=text_tags,
                tags_to_filter=text_tags_to_filter,
                transformer_model=text_model,
                max_text_length=max_text_length)

        return cls(continuous_layer=continuous_layer,
                   categorical_layer=categorical_layer,
                   text_embedding_layer=text_model,
                   aggregation=aggregation,
                   **kwargs)

    def compute_output_shape(self, input_shapes):
        output_shapes = {}
        for in_layer in self.to_apply:
            output_shapes.update(in_layer.compute_output_shape(input_shapes))

        return super().compute_output_shape(output_shapes)


class DenseFeatures(tf.keras.layers.Layer):
    """
    Layer which maps a dictionary of input tensors to a dense, continuous
    vector digestible by a neural network. Meant to reproduce the API exposed
    by `tf.keras.layers.DenseFeatures` while reducing overhead for the
    case of one-hot categorical and scalar numeric features.

    Uses TensorFlow `feature_column` to represent inputs to the layer, but
    does not perform any preprocessing associated with those columns. As such,
    it should only be passed `numeric_column` objects and their subclasses,
    `embedding_column` and `indicator_column`. Preprocessing functionality should
    be moved to NVTabular.

    For multi-hot categorical or vector continuous data, represent the data for
    a feature with a dictionary entry `"<feature_name>__values"` corresponding
    to the flattened array of all values in the batch. For multi-hot categorical
    data, there should be a corresponding `"<feature_name>__nnzs"` entry that
    describes how many categories are present in each sample (and so has length
    `batch_size`).

    Note that categorical columns should be wrapped in embedding or
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
        x = DenseFeatures([column_a, column_b_embedding])(inputs)

    Parameters
    ----------
    feature_columns : list of `tf.feature_column`
        feature columns describing the inputs to the layer
    aggregation : str in ("concat", "stack")
        how to combine the embeddings from multiple features
    """

    def __init__(self, feature_columns, aggregation="concat", name=None, **kwargs):
        # sort feature columns to make layer independent of column order
        feature_columns = _sort_columns(feature_columns)
        _validate_dense_feature_columns(feature_columns)

        if aggregation == "stack":
            _validate_stack_dimensions(feature_columns)
        elif aggregation != "concat":
            raise ValueError(
                "Unrecognized aggregation {}, must be stack or concat".format(aggregation)
            )

        self.feature_columns = feature_columns
        self.aggregation = aggregation
        super(DenseFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
        assert all(shape[1] == 1 for shape in input_shapes.values() if not isinstance(shape, tuple))
        assert all(shape[0][1] == 1 for shape in input_shapes.values() if isinstance(shape, tuple))
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
                x = _handle_continuous_feature(inputs, feature_column)
                features.append(x)
            else:
                feature_name = feature_column.categorical_column.name
                table = self.embedding_tables[feature_name]
                combiner = getattr(feature_column, "combiner", "sum")
                embeddings = _categorical_embedding_lookup(table, inputs, feature_name, combiner)
                features.append(embeddings)

        if self.aggregation == "stack":
            return tf.stack(features, axis=1)
        return tf.concat(features, axis=1)

    def compute_output_shape(self, input_shapes):
        input_shape = list(input_shapes.values())[0]
        if self.aggregation == "concat":
            output_dim = len(self.numeric_features) + sum(
                [shape[-1] for shape in self.embedding_shapes.values()]
            )
            return (input_shape[0], output_dim)
        else:
            embedding_dim = list(self.embedding_shapes.values())[0]
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
class LinearFeatures(tf.keras.layers.Layer):
    """
    Layer which implements a linear combination of one-hot categorical
    and scalar numeric features. Based on the "wide" branch of the Wide & Deep
    network architecture.

    Uses TensorFlow ``feature_column``s to represent inputs to the layer, but
    does not perform any preprocessing associated with those columns. As such,
    it should only be passed ``numeric_column`` and
    ``categorical_column_with_identity``. Preprocessing functionality should
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
        super(LinearFeatures, self).__init__(name=name, **kwargs)

    def build(self, input_shapes):
        assert all(shape[1] == 1 for shape in input_shapes.values() if not isinstance(shape, tuple))
        assert all(shape[0][1] == 1 for shape in input_shapes.values() if isinstance(shape, tuple))
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
                numeric_inputs.append(_handle_continuous_feature(inputs, feature_column))
            else:
                table = self.embedding_tables[feature_column.key]
                embeddings = _categorical_embedding_lookup(table, inputs, feature_column.key, "sum")
                x = x + embeddings

        if len(numeric_inputs) > 0:
            numerics = tf.concat(numeric_inputs, axis=1)
            x = x + tf.matmul(numerics, self.embedding_tables["numeric"])
        return x

    def compute_output_shape(self, input_shapes):
        batch_size = list(input_shapes.values())[0].shape[0]
        return (batch_size, 1)

    def get_config(self):
        return {
            "feature_columns": self.feature_columns,
        }
