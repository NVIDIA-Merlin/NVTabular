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

from merlin_models.tf.layers import DenseFeatures, DotProductInteraction, LinearFeatures
from merlin_models.tf.models import model_utils


class DeepFM(tf.keras.Model):
    """
    https://arxiv.org/pdf/1703.04247.pdf
    Note that this paper doesn't specify how continuous features are
    handled during the FM portion, so I'm just going to exclude them
    and only use them for the deep portion. There's probably better
    ways to handle this, but since this is meant to be a verbatim
    implementation of the paper I'm going to take the simplest
    route possible
    """

    def __init__(
        self,
        continuous_columns,
        categorical_columns,
        embedding_dims,
        hidden_dims=None,
        use_wide=False,
    ):
        super().__init__()
        channels = self.channels(continuous_columns, categorical_columns, embedding_dims, use_wide)

        hidden_dims = hidden_dims or []

        self.deep_embedding_layer = DenseFeatures(channels["deep"])
        self.fm_embedding_layer = DenseFeatures(channels["fm"])

        # Deep channel
        self.deep_input_layer = tf.keras.layers.Concatenate(axis=1)
        self.deep_final_layer = tf.keras.layers.Dense(1, activation="linear")

        self.deep_hidden_layers = []
        for dim in hidden_dims:
            self.deep_hidden_layers.append(tf.keras.layers.Dense(dim, activation="relu"))
            # + batchnorm, dropout, whatever...

        # FM channel
        fm_shape = (len(channels["fm"]), embedding_dims)
        self.fm_reshape_layer = tf.keras.layers.Reshape(fm_shape)
        self.fm_interaction_layer = DotProductInteraction()

        # Optional Wide channel
        if "wide" in channels:
            self.wide_linear_layer = LinearFeatures(channels["wide"])
        else:
            self.wide_linear_layer = None

        # Channel combiner
        self.combiner_concat = tf.keras.layers.Concatenate(axis=1)
        self.combiner_reduce_sum = tf.keras.layers.Lambda(
            lambda x: tf.reduce_sum(x, axis=1, keepdims=True)
        )
        self.combiner_activation = tf.keras.layers.Activation("sigmoid")

    def channels(self, continuous_columns, categorical_columns, embedding_dim, use_wide=False):
        embedding_columns = model_utils.get_embedding_columns(categorical_columns, embedding_dim)

        channels = {"fm": embedding_columns, "deep": continuous_columns}

        if use_wide:
            channels["wide"] = categorical_columns + continuous_columns

        return channels

    def call(self, inputs, training=False):
        fm_embeddings = self.fm_embedding_layer(inputs["fm"])
        deep_embeddings = self.deep_embedding_layer(inputs["deep"])

        # Deep channel
        deep_x = self.deep_input_layer([fm_embeddings, deep_embeddings])
        for layer in self.deep_hidden_layers:
            deep_x = layer(deep_x)
        deep_x = self.deep_final_layer(deep_x)

        # FM channel
        fm_embeddings = self.fm_reshape_layer(fm_embeddings)
        fm_x = self.fm_interaction_layer(fm_embeddings)

        # Optional Wide channel
        activation_inputs = [fm_x, deep_x]
        if self.wide_linear_layer:
            wide_x = self.wide_linear_layer(inputs["wide"])
            activation_inputs.append(wide_x)

        # Combine, Sum, Activate
        # note that we can't just add since `fm_x` has dim
        # k(k-1)/2, where k is the number of categorical features
        combined_x = self.combiner_concat(activation_inputs)
        combined_x = self.combiner_reduce_sum(combined_x)
        combined_x = self.combiner_activation(combined_x)

        return combined_x
