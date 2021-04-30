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

import os

import tensorflow as tf

from nvtabular.framework_utils.tensorflow import layers
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.ops import get_embedding_sizes


def train_tensorflow(workflow, out_path, cats, conts, labels, batch_size):
    # Get embeddings from workflow
    embeddings = get_embedding_sizes(workflow)
    for key in embeddings:

        embeddings[key] = (
            embeddings[key][0],
            min(16, embeddings[key][1]),
        )

    # Set paths and dataloaders
    train_path = os.path.join(out_path, "train/")
    valid_path = os.path.join(out_path, "valid/")

    train_dataset_tf = KerasSequenceLoader(
        train_path,
        batch_size=batch_size,
        label_names=labels,
        cat_names=cats,
        cont_names=conts,
        engine="parquet",
        shuffle=True,
        buffer_size=0.06,
        parts_per_chunk=1,
    )

    valid_dataset_tf = KerasSequenceLoader(
        valid_path,
        batch_size=batch_size,
        label_names=labels,
        cat_names=cats,
        cont_names=conts,
        engine="parquet",
        shuffle=False,
        buffer_size=0.06,
        parts_per_chunk=1,
    )

    inputs = {}  # tf.keras.Input placeholders for each feature to be used
    emb_layers = []  # output of all embedding layers, which will be concatenated
    num_layers = []  # output of numerical layers

    for col in cats:
        inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))

    for col in conts:
        inputs[col] = tf.keras.Input(name=col, dtype=tf.float32, shape=(1,))

    for col in cats:
        emb_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, embeddings[col][0]  # Input dimension (vocab size)
                ),
                embeddings[col][1],  # Embedding output dimension
            )
        )

    for col in conts:
        num_layers.append(tf.feature_column.numeric_column(col))

    emb_layer = layers.DenseFeatures(emb_layers)
    x_emb_output = emb_layer(inputs)

    x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile("sgd", "binary_crossentropy")

    tf.keras.utils.plot_model(model)

    validation_callback = KerasSequenceValidater(valid_dataset_tf)

    model.fit(train_dataset_tf, callbacks=[validation_callback], epochs=1)

    model.save(os.path.join(out_path, "model.savedmodel"))
