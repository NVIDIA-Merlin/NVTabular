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

from merlin_models.tf.layers import DenseFeatures


class YouTubeDNN(tf.keras.Model):
    """
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf
    See model architecture diagram in Figure 3
    """

    def __init__(self, continuous_columns, categorical_columns, embedding_dims, hidden_dims=None):

        super().__init__()

        hidden_dims = hidden_dims or []

        channels = self.channels(continuous_columns, categorical_columns, embedding_dims)

        self.input_layer = DenseFeatures(channels["categorical"] + channels["continuous"])

        self.hidden_layers = []
        for dim in hidden_dims:
            self.hidden_layers.append(tf.keras.layers.Dense(dim, activation="relu"))

    def channels(self, continuous_columns, categorical_columns, embedding_dims):
        if not isinstance(embedding_dims, dict):
            embedding_dims = {col.name: embedding_dims for col in categorical_columns}

        embedding_columns = [
            tf.feature_column.embedding_column(col, embedding_dims[col.name])
            for col in categorical_columns
        ]

        return {"categorical": embedding_columns, "continuous": continuous_columns}

    def call(self, inputs, training=False):
        x = self.input_layer({**inputs["categorical"], **inputs["continuous"]})
        for layer in self.hidden_layers:
            x = layer(x)
        return x
