import math
from typing import Dict, Optional

import tensorflow as tf
from nvtabular import ColumnGroup
from nvtabular.ops import get_embedding_sizes
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.tpu.tpu_embedding_v2_utils import (
    FeatureConfig,
    TableConfig,
)

from merlin_models.tf.tabular import TabularLayer, AsSparseLayer, FilterFeatures


class EmbeddingFeatures(TabularLayer):
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
                          infer_embedding_sizes=True, combiner="mean", tags=None, tags_to_filter=None,
                          **kwargs) -> Optional["EmbeddingFeatures"]:
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

        if not feature_config:
            return None

        return cls(feature_config, **kwargs)