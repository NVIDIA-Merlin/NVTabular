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

from merlin_models.tf.layers import DenseFeatures, LinearFeatures, XDeepFmOuterProduct
from merlin_models.tf.models import model_utils


class xDeepFM(tf.keras.Model):
    """
    https://arxiv.org/pdf/1803.05170.pdf
    """

    def __init__(
        self,
        continuous_columns,
        categorical_columns,
        embedding_dims,
        deep_hidden_dims=None,
        cin_hidden_dims=None,
        use_wide=False,
    ):
        super().__init__()

        deep_hidden_dims = deep_hidden_dims or []
        cin_hidden_dims = cin_hidden_dims or []

        channels = self.channels(continuous_columns, categorical_columns, embedding_dims, use_wide)

        self.categorical_embedding_layer = DenseFeatures(channels["CIN"])
        self.continuous_embedding_layer = DenseFeatures(channels["deep"])

        # Deep channel
        self.deep_input_layer = tf.keras.layers.Concatenate(axis=1)
        self.deep_final_layer = tf.keras.layers.Dense(1, activation="linear")

        self.deep_hidden_layers = []

        for dim in deep_hidden_dims:
            self.deep_hidden_layers.append(tf.keras.layers.Dense(dim, activation="relu"))
            # + batchnorm, dropout, whatever...

        # Compressed Interaction Network channel
        num_cin_inputs = len(channels["CIN"])
        self.cin_reshape_layer = tf.keras.layers.Reshape((num_cin_inputs, embedding_dims))

        self.cin_sum_pool_layer = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=2))

        self.cin_hidden_layers = []
        for dim in cin_hidden_dims:
            self.cin_hidden_layers.append(XDeepFmOuterProduct(dim))

        self.cin_final_layer = tf.keras.layers.Concatenate(axis=1)

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

        # not really clear to me how to use numeric columns in CIN so will
        # only feed them to deep channel
        channels = {"CIN": embedding_columns, "deep": continuous_columns}

        if use_wide:
            channels["wide"] = continuous_columns + categorical_columns

        return channels

    def call(self, inputs, training=False):
        categorical_embeddings = self.categorical_embedding_layer(inputs["CIN"])
        continuous_embeddings = self.continuous_embedding_layer(inputs["deep"])

        deep_x = self.deep_input_layer([categorical_embeddings, continuous_embeddings])
        for layer in self.deep_hidden_layers:
            deep_x = layer(deep_x)
        deep_x = self.deep_final_layer(deep_x)

        cin_x0 = self.cin_reshape_layer(categorical_embeddings)
        cin_x = cin_x0

        pooled_outputs = []
        for cin_hidden_layer in self.cin_hidden_layers:
            cin_x = cin_hidden_layer([cin_x, cin_x0])
            pooled_outputs.append(self.cin_sum_pool_layer(cin_x))
        cin_x = self.cin_final_layer(pooled_outputs)

        activation_inputs = [cin_x, deep_x]
        if self.wide_linear_layer:
            wide_x = self.wide_linear_layer(inputs["wide"])
            activation_inputs.append(wide_x)

        combined_x = self.combiner_concat(activation_inputs)
        combined_x = self.combiner_reduce_sum(combined_x)
        combined_x = self.combiner_activation(combined_x)

        return combined_x
