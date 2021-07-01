from typing import Optional, Union, List

import tensorflow as tf

from nvtabular.column_group import ColumnGroup
from nvtabular.framework_utils.tensorflow import EmbeddingsLayer
from nvtabular.framework_utils.tensorflow.blocks.base import Block, BlockType
from nvtabular.framework_utils.tensorflow.features import TabularLayer


class DLRMBlock(Block):
    def __init__(self,
                 continuous_features: Union[List[str], ColumnGroup, TabularLayer],
                 embedding_layer: EmbeddingsLayer,
                 bottom_mlp: BlockType,
                 top_mlp: Optional[BlockType] = None,
                 interaction_layer: Optional[tf.keras.layers.Layer] = None,
                 trainable=True,
                 name=None,
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.continuous_features = continuous_features
        self.embedding_layer = embedding_layer
        self.embedding_layer.aggregation = "stack"
        self.bottom_mlp = bottom_mlp
        self.top_mlp = top_mlp

        if isinstance(continuous_features, TabularLayer):
            self.con_input_layer = continuous_features
        elif isinstance(continuous_features, ColumnGroup):
            self.con_input_layer = TabularLayer.from_column_group(continuous_features, aggregation="concat")
        elif isinstance(continuous_features, list):
            self.con_input_layer = TabularLayer.from_features(continuous_features, aggregation="concat")

        to_tabular = tf.keras.layers.Lambda(lambda x: dict(continuous=tf.expand_dims(x, 1)))
        self.continuous_embedding = self.con_input_layer >> bottom_mlp >> to_tabular

        from nvtabular.framework_utils.tensorflow.layers import DotProductInteraction
        self.interaction_layer = interaction_layer or DotProductInteraction()

    @classmethod
    def from_column_group(cls,
                          column_group: ColumnGroup,
                          bottom_mlp: BlockType,
                          top_mlp: Optional[BlockType] = None,
                          **kwargs):
        embedding_layer = EmbeddingsLayer.from_column_group(
            column_group.categorical_column_group,
            infer_embedding_sizes=False,
            default_embedding_dim=bottom_mlp.layers[-1].units,
            aggregation="stack"
        )

        continuous_features = TabularLayer.from_column_group(
            column_group.continuous_column_group,
            aggregation="concat"
        )

        return cls(continuous_features, embedding_layer, bottom_mlp, top_mlp=top_mlp, **kwargs)

    def call(self, inputs, **kwargs):
        stacked = self.embedding_layer(inputs, merge_with=self.continuous_embedding)
        interactions = self.interaction_layer(stacked)

        return interactions if not self.top_mlp else self.top_mlp(interactions)
