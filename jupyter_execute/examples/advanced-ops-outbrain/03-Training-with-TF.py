#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
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
# ==============================================================================


# # NVTabular demo on Outbrain Data

# ## Overview

# In this notebook we train TF Wide & Deep Learning framework using [Kaggle Outbrain dataset](https://www.kaggle.com/c/outbrain-click-prediction). In that competition, ‘Kagglers’ were challenged to predict on which ads and other forms of sponsored content its global users would click. 
# 
# [Wide & Deep Learning](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html) refers to a class of networks that use the output of two parts working in parallel - wide model and deep model - to make predictions using categorical and continuous inputs. The wide model is a generalized linear model of features together with their transforms. The deep model in this notebook is a series of three hidden MLP layers of [1024, 512, 256] neurons each beginning with a dense embedding of features. 
# 
# **Learning objectives**<br><br>
# This notebook explains, how to use the NVTabular dataloader to accelerate TensorFlow training.
# 
# - Use NVTabular dataloader with TensorFlow Keras model
# - Training Wide&Deep model with NVTabular dataloader in TensorFlow

# In[2]:


import os
import tensorflow as tf

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
# os.environ['TF_MEMORY_ALLOCATION'] = "0.8" # fraction of free memory

from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.framework_utils.tensorflow import layers


# In[3]:


import glob

import nvtabular as nvt


# First, we set where the parquet datasets were saved once processed (OUTPUT_BUCKET_FOLDER).

# In[4]:


OUTPUT_BUCKET_FOLDER = os.environ.get("OUTPUT_DATA_DIR", "./outbrain-preprocessed/")


# In the previous notebook, we used NVTabular for ETL and stored the workflow to disk. We can load the NVTabular workflow to extract important metadata for our training pipeline.

# In[5]:


workflow = nvt.Workflow.load(os.path.join(OUTPUT_BUCKET_FOLDER, "workflow"))


# EMBEDDING_TABLE_SHAPES defines the size of the embedding tables that our model will use to map categorical outputs from NVTabular into numeric dense inputs.

# In[6]:


from nvtabular.ops import get_embedding_sizes

EMBEDDING_TABLE_SHAPES = {
    column: (shape[0], min(shape[1], 16)) for column, shape in get_embedding_sizes(workflow).items()
}
EMBEDDING_TABLE_SHAPES


# We select define categorical and numerical features that are processed and generated via the NVTabular workflow to train our W&D TF model.

# In[7]:


CATEGORICAL_COLUMNS = [
    "geo_location",
    "geo_location_country",
    "geo_location_state",
    "ad_id",
    "document_id",
    "platform",
    "document_id_promo",
    "campaign_id",
    "advertiser_id",
    "source_id",
    "publisher_id",
    "source_id_promo",
    "publisher_id_promo",
]


# In[8]:


NUMERIC_COLUMNS = [
    "document_id_document_id_promo_sim_categories",
    "document_id_document_id_promo_sim_topics",
    "document_id_document_id_promo_sim_entities",
    "publish_time_since_published",
    "publish_time_promo_since_published",
    "TE_ad_id_clicked",
    "TE_document_id_promo_clicked",
    "TE_campaign_id_clicked",
    "TE_advertiser_id_clicked",
    "TE_source_id_clicked",
    "TE_publisher_id_clicked",
]


# ## Training a TF W&D Model

# We create tensorflow feature columns corresponding to each feature of the model input. If you're using NVTabular with TensorFlow feature_columns, you should only be using `tf.feature_column.categorical_column_with_identity` for categorical features, since any other transformation (categorification and/or hashing) should be handled in NVTabular on the GPU. This feature column is passed to the wide portion of the model. If a categorical column corresponds to an embedding table, it is wrapped with an embedding_column feature_column, if it does not correspond to an embedding table, it is wrapped as an indicator column. The wrapped column is passed to the deep portion of the model. Continuous columns are passed to both the wide and deep portions of the model after being encapsulated as a `numeric_column`.

# In[9]:


def get_feature_columns():
    wide_columns, deep_columns = [], []

    for column_name in CATEGORICAL_COLUMNS:
        if (
            column_name in EMBEDDING_TABLE_SHAPES
        ):  # Changing hashing to identity + adding modulo to dataloader
            categorical_column = tf.feature_column.categorical_column_with_identity(
                column_name, num_buckets=EMBEDDING_TABLE_SHAPES[column_name][0]
            )
        else:
            raise ValueError(f"Unexpected categorical column found {column_name}")

        if column_name in EMBEDDING_TABLE_SHAPES:
            wrapped_column = tf.feature_column.embedding_column(
                categorical_column,
                dimension=EMBEDDING_TABLE_SHAPES[column_name][1],
                combiner="mean",
            )
        else:
            wrapped_column = tf.feature_column.indicator_column(categorical_column)

        wide_columns.append(categorical_column)
        deep_columns.append(wrapped_column)

    numerics = [
        tf.feature_column.numeric_column(column_name, shape=(1,), dtype=tf.float32)
        for column_name in NUMERIC_COLUMNS
    ]

    wide_columns.extend(numerics)
    deep_columns.extend(numerics)

    return wide_columns, deep_columns


# Next, we define the layer shape and dropout probability for the deep portion of the model.

# In[10]:


deep_hidden_units = [1024, 512, 256]
deep_dropout = 0.1


# An input is created for each feature column, with a datatype of either tf.float32 for continuous values, or tf.int32 for categorical values. To implement the wide model, for categorical inputs, we embed them to a dimension of one, and sum them with the results of applying a dense layer with output dimension one, effectively weighting and summing each of the inputs. For the deep model, we embed our categorical columns according to the feature columns we defined earlier, and concatenate the newly dense features with our dense continuous features, which we pass to our deep model, which by default is a 5 layer MLP with internal dimension of 1024 neurons for each layer. 

# In[11]:


wide_columns, deep_columns = get_feature_columns()

wide_weighted_outputs = []  # a list of (batch_size, 1) contributors to the linear weighted sum
numeric_dense_inputs = []  # NumericColumn inputs; to be concatenated and then fed to a dense layer
wide_columns_dict = {}  # key : column
deep_columns_dict = {}  # key : column
features = {}  # tf.keras.Input placeholders for each feature to be used

# construct input placeholders for wide features
for col in wide_columns:
    features[col.key] = tf.keras.Input(
        shape=(1,),
        batch_size=None,
        name=col.key,
        dtype=tf.float32 if col.key in NUMERIC_COLUMNS else tf.int32,
        sparse=False,
    )
    wide_columns_dict[col.key] = col
for col in deep_columns:
    is_embedding_column = "key" not in dir(col)
    key = col.categorical_column.key if is_embedding_column else col.key

    if key not in features:
        features[key] = tf.keras.Input(
            shape=(1,),
            batch_size=None,
            name=key,
            dtype=tf.float32 if col.key in NUMERIC_COLUMNS else tf.int32,
            sparse=False,
        )
    deep_columns_dict[key] = col

for key in wide_columns_dict:
    if key in EMBEDDING_TABLE_SHAPES:
        wide_weighted_outputs.append(
            tf.keras.layers.Flatten()(
                tf.keras.layers.Embedding(EMBEDDING_TABLE_SHAPES[key][0], 1, input_length=1)(
                    features[key]
                )
            )
        )
    else:
        numeric_dense_inputs.append(features[key])

categorical_output_contrib = tf.keras.layers.add(wide_weighted_outputs, name="categorical_output")
numeric_dense_tensor = tf.keras.layers.concatenate(numeric_dense_inputs, name="numeric_dense")
deep_columns = list(deep_columns_dict.values())

dnn = layers.DenseFeatures(deep_columns, name="deep_embedded")(features)
for unit_size in deep_hidden_units:
    dnn = tf.keras.layers.Dense(units=unit_size, activation="relu")(dnn)
    dnn = tf.keras.layers.Dropout(rate=deep_dropout)(dnn)
    dnn = tf.keras.layers.BatchNormalization(momentum=0.999)(dnn)
dnn = tf.keras.layers.Dense(units=1)(dnn)
dnn_model = tf.keras.Model(inputs=features, outputs=dnn)
linear_output = categorical_output_contrib + tf.keras.layers.Dense(1)(numeric_dense_tensor)

linear_model = tf.keras.Model(inputs=features, outputs=linear_output)

wide_and_deep_model = tf.keras.experimental.WideDeepModel(
    linear_model, dnn_model, activation="sigmoid"
)


# We define the datasets that will be used to ingest data into our model. In this case, the NVTabular dataloaders take a set of parquet files generated by NVTabular as input, and are capable of accelerated throughput. The [KerasSequenceLoader](https://github.com/NVIDIA/NVTabular/blob/9aa70caa1dfb5d2fd694cad535def1e470d37b29/nvtabular/loader/tensorflow.py#L89) manages shuffling by loading in chunks of data from different parts of the full dataset, concatenating them and then shuffling, then iterating through this super-chunk sequentially in batches. The number of "parts" of the dataset that get sample, or "partitions", is controlled by the <i>parts_per_chunk</i> parameter, while the size of each one of these parts is controlled by the <i>buffer_size</i> parameter, which refers to a fraction of available GPU memory. Using more chunks leads to better randomness, especially at the epoch level where physically disparate samples can be brought into the same batch, but can impact throughput if you use too many.
# 
# The validation process gets slightly complicated by the fact that <i>model.fit</i> doesn't accept Keras Sequence objects as validation data. To support this, we also define a [KerasSequenceValidater](https://github.com/NVIDIA/NVTabular/blob/9aa70caa1dfb5d2fd694cad535def1e470d37b29/nvtabular/loader/tensorflow.py#L351), a lightweight Keras callback to handle validation.

# Now that our data is preprocessed and saved out, we can leverage datasets to read through the preprocessed parquet files in an online fashion to train neural networks.

# In[12]:


TRAIN_PATHS = sorted(glob.glob(os.path.join(OUTPUT_BUCKET_FOLDER, "train/*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(OUTPUT_BUCKET_FOLDER, "valid/*.parquet")))


# In[13]:


train_dataset_tf = KerasSequenceLoader(
    TRAIN_PATHS,  # you could also use a glob pattern
    batch_size=131072,
    label_names=["clicked"],
    cat_names=CATEGORICAL_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.06,  # how many batches to load at once
    parts_per_chunk=1,
)

valid_dataset_tf = KerasSequenceLoader(
    VALID_PATHS,  # you could also use a glob pattern
    batch_size=131072,
    label_names=["clicked"],
    cat_names=CATEGORICAL_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=False,
    buffer_size=0.06,
    parts_per_chunk=1,
)
validation_callback = KerasSequenceValidater(valid_dataset_tf)


# The wide portion of the model is optimized using the <i>Follow The Regularized Leader (FTRL)</i> algorithm, while the deep portion of the model is optimized using <i>Adam</i> optimizer.

# In[14]:


wide_optimizer = tf.keras.optimizers.Ftrl(
    learning_rate=0.1,
)

deep_optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)


# Finally, we compile our model with our dual optimizers and binary cross-entropy loss, and train our model for 10 epochs.

# In[15]:


wide_and_deep_model.compile(
    optimizer=[wide_optimizer, deep_optimizer],
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()],
    experimental_run_tf_function=False,
)
history = wide_and_deep_model.fit(train_dataset_tf, callbacks=[validation_callback], epochs=5)

