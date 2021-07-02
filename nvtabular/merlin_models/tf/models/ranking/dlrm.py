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

import tensorflow as tf

from merlin_models.tf.layers import DenseFeatures, DotProductInteraction
from merlin_models.tf.models import model_utils


class DLRM(tf.keras.Model):
    """
    https://arxiv.org/pdf/1906.00091.pdf
    See model description at the bottom of page 3
    """

    def __init__(
        self,
        continuous_columns,
        categorical_columns,
        embedding_dims,
        dense_hidden_dims=None,
        combiner_hidden_dims=None,
    ):
        super().__init__()

        dense_hidden_dims = dense_hidden_dims or []
        combiner_hidden_dims = combiner_hidden_dims or []

        channels = self.channels(
            continuous_columns,
            categorical_columns,
            embedding_dims,
        )

        self.fm_features_layer = DenseFeatures(channels["fm"], aggregation="stack")
        self.dense_features_layer = DenseFeatures(channels["dense"], aggregation="concat")

        # Dense channel (bottom MLP)
        self.dense_hidden_layers = []
        for dim in dense_hidden_dims:
            self.dense_hidden_layers.append(tf.keras.layers.Dense(dim, activation="relu"))
            # + batchnorm, dropout, whatever...

        self.dense_final_layer = tf.keras.layers.Dense(embedding_dims, activation="relu")

        # FM channel
        self.fm_dense_input_layer = tf.keras.layers.Reshape((1, embedding_dims))
        self.fm_concat_layer = tf.keras.layers.Concatenate(axis=1)
        self.fm_interaction_layer = DotProductInteraction()

        # Combiner (top MLP)
        self.combiner_concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.combiner_final_layer = tf.keras.layers.Dense(1, activation="sigmoid")

        self.combiner_hidden_layers = []
        for dim in combiner_hidden_dims:
            self.combiner_hidden_layers.append(tf.keras.layers.Dense(dim, activation="relu"))
            # + batchnorm, dropout, whatever...

    def channels(self, continuous_columns, categorical_columns, embedding_dim):
        embedding_columns = model_utils.get_embedding_columns(categorical_columns, embedding_dim)
        return {"dense": continuous_columns, "fm": embedding_columns}

    def call(self, inputs, training=False):
        fm = self.fm_features_layer(inputs["fm"])
        dense = self.dense_features_layer(inputs["dense"])

        # Dense channel (bottom MLP)
        for layer in self.dense_hidden_layers:
            dense = layer(dense)
        dense = self.dense_final_layer(dense)

        # FM channel
        dense_fm = self.fm_dense_input_layer(dense)
        fm = self.fm_concat_layer([fm, dense_fm])
        fm = self.fm_interaction_layer(fm)

        # Combiner (top MLP)
        combined_x = self.combiner_concat_layer([fm, dense])
        for layer in self.combiner_hidden_layers:
            combined_x = layer(combined_x)
        combined_x = self.combiner_final_layer(combined_x)

        return combined_x
