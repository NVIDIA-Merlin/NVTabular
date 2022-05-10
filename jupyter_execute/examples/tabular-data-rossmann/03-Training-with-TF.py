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


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # NVTabular demo on Rossmann data - TensorFlow
# 
# ## Overview
# 
# NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems.  It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.

# ### Learning objectives
# 
# In the previous notebooks ([01-Download-Convert.ipynb](https://github.com/NVIDIA/NVTabular/blob/main/examples/99-applying-to-other-tabular-data-problems-rossmann/01-Download-Convert.ipynb) and [02-ETL-with-NVTabular.ipynb](https://github.com/NVIDIA/NVTabular/blob/main/examples/99-applying-to-other-tabular-data-problems-rossmann/02-ETL-with-NVTabular.ipynb)), we downloaded, preprocessed and created features for the dataset. Now, we are ready to train our deep learning model on the dataset. In this notebook, we use **TensorFlow** with the NVTabular data loader for TensorFlow to accelereate the training pipeline.

# In[2]:


import os
import math
import json
import glob


# ## Loading NVTabular workflow
# This time, we only need to define our data directories. We can load the data schema from the NVTabular workflow.

# In[3]:


DATA_DIR = os.environ.get("INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/data/"))
PREPROCESS_DIR = os.path.join(DATA_DIR, "ross_pre/")
PREPROCESS_DIR_TRAIN = os.path.join(PREPROCESS_DIR, "train")
PREPROCESS_DIR_VALID = os.path.join(PREPROCESS_DIR, "valid")


# What files are available to train on in our directories?

# In[4]:


get_ipython().system('ls $PREPROCESS_DIR')


# In[5]:


get_ipython().system('ls $PREPROCESS_DIR_TRAIN')


# In[6]:


get_ipython().system('ls $PREPROCESS_DIR_VALID')


# We load the data schema and statistic information from `stats.json`. We created the file in the previous notebook `rossmann-store-sales-feature-engineering`.

# In[7]:


stats = json.load(open(PREPROCESS_DIR + "/stats.json", "r"))


# In[8]:


CATEGORICAL_COLUMNS = stats["CATEGORICAL_COLUMNS"]
CONTINUOUS_COLUMNS = stats["CONTINUOUS_COLUMNS"]
LABEL_COLUMNS = stats["LABEL_COLUMNS"]
COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + LABEL_COLUMNS


# The embedding table shows the cardinality of each categorical variable along with its associated embedding size. Each entry is of the form `(cardinality, embedding_size)`.

# In[9]:


EMBEDDING_TABLE_SHAPES = stats["EMBEDDING_TABLE_SHAPES"]
EMBEDDING_TABLE_SHAPES


# ## Training a Network
# 
# Now that our data is preprocessed and saved out, we can leverage `dataset`s to read through the preprocessed parquet files in an online fashion to train neural networks.
# 
# We'll start by setting some universal hyperparameters for our model and optimizer. These settings will be the same across all of the frameworks that we explore in the different notebooks.

# If you're interested in contributing to NVTabular, feel free to take this challenge on and submit a pull request if successful. 12% RMSPE is achievable using the Novograd optimizer, but we know of no Novograd implementation for TensorFlow that supports sparse gradients, and so we are not including that solution below.

# In[10]:


EMBEDDING_DROPOUT_RATE = 0.04
DROPOUT_RATES = [0.001, 0.01]
HIDDEN_DIMS = [1000, 500]
BATCH_SIZE = 65536
LEARNING_RATE = 0.001
EPOCHS = 25

# TODO: Calculate on the fly rather than recalling from previous analysis.
MAX_SALES_IN_TRAINING_SET = 38722.0
MAX_LOG_SALES_PREDICTION = 1.2 * math.log(MAX_SALES_IN_TRAINING_SET + 1.0)

TRAIN_PATHS = sorted(glob.glob(os.path.join(PREPROCESS_DIR_TRAIN, "*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(PREPROCESS_DIR_VALID, "*.parquet")))


# ## TensorFlow
# <a id="TensorFlow"></a>
# 

# ### TensorFlow: Preparing Datasets
# 
# `KerasSequenceLoader` wraps a lightweight iterator around a `dataset` object to handle chunking, shuffling, and application of any workflows (which can be applied online as a preprocessing step). For column names, can use either a list of string names or a list of TensorFlow `feature_columns` that will be used to feed the network

# In[11]:


import tensorflow as tf

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# it's too late and TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "8192"  # explicit MB
os.environ["TF_MEMORY_ALLOCATION"] = "0.5"  # fraction of free memory
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater


# cheap wrapper to keep things some semblance of neat
def make_categorical_embedding_column(name, dictionary_size, embedding_dim):
    return tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_identity(name, dictionary_size), embedding_dim
    )


# instantiate our columns
categorical_columns = [
    make_categorical_embedding_column(name, *EMBEDDING_TABLE_SHAPES[name])
    for name in CATEGORICAL_COLUMNS
]
continuous_columns = [tf.feature_column.numeric_column(name, (1,)) for name in CONTINUOUS_COLUMNS]

# feed them to our datasets
train_dataset = KerasSequenceLoader(
    TRAIN_PATHS,  # you could also use a glob pattern
    feature_columns=categorical_columns + continuous_columns,
    batch_size=BATCH_SIZE,
    label_names=LABEL_COLUMNS,
    shuffle=True,
    buffer_size=0.06,  # amount of data, as a fraction of GPU memory, to load at once
)

valid_dataset = KerasSequenceLoader(
    VALID_PATHS,  # you could also use a glob pattern
    feature_columns=categorical_columns + continuous_columns,
    batch_size=BATCH_SIZE * 4,
    label_names=LABEL_COLUMNS,
    shuffle=False,
    buffer_size=0.06,  # amount of data, as a fraction of GPU memory, to load at once
)


# ### TensorFlow: Defining a Model
# 
# Using Keras, we can define the layers of our model and their parameters explicitly. Here, for the sake of consistency, we'll mimic fast.ai's [TabularModel](https://docs.fast.ai/tabular.learner.html).

# In[12]:


# DenseFeatures layer needs a dictionary of {feature_name: input}
categorical_inputs = {}
for column_name in CATEGORICAL_COLUMNS:
    categorical_inputs[column_name] = tf.keras.Input(name=column_name, shape=(1,), dtype=tf.int64)
categorical_embedding_layer = tf.keras.layers.DenseFeatures(categorical_columns)
categorical_x = categorical_embedding_layer(categorical_inputs)
categorical_x = tf.keras.layers.Dropout(EMBEDDING_DROPOUT_RATE)(categorical_x)

# Just concatenating continuous, so can use a list
continuous_inputs = []
for column_name in CONTINUOUS_COLUMNS:
    continuous_inputs.append(tf.keras.Input(name=column_name, shape=(1,), dtype=tf.float64))
continuous_embedding_layer = tf.keras.layers.Concatenate(axis=1)
continuous_x = continuous_embedding_layer(continuous_inputs)
continuous_x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(continuous_x)

# concatenate and build MLP
x = tf.keras.layers.Concatenate(axis=1)([categorical_x, continuous_x])
for dim, dropout_rate in zip(HIDDEN_DIMS, DROPOUT_RATES):
    x = tf.keras.layers.Dense(dim, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
x = tf.keras.layers.Dense(1, activation="linear")(x)

# TODO: Initialize model weights to fix saturation issues.
# For now, we'll just scale the output of our model directly before
# hitting the sigmoid.
x = 0.1 * x

x = MAX_LOG_SALES_PREDICTION * tf.keras.activations.sigmoid(x)

# combine all our inputs into a single list
# (note that you can still use .fit, .predict, etc. on a dict
# that maps input tensor names to input values)
inputs = list(categorical_inputs.values()) + continuous_inputs
tf_model = tf.keras.Model(inputs=inputs, outputs=x)


# ### TensorFlow: Training

# In[13]:


def rmspe_tf(y_true, y_pred):
    # map back into "true" space by undoing transform
    y_true = tf.exp(y_true) - 1
    y_pred = tf.exp(y_pred) - 1

    percent_error = (y_true - y_pred) / y_true
    return tf.sqrt(tf.reduce_mean(percent_error ** 2))


# In[14]:


get_ipython().run_cell_magic('time', '', 'from time import time\n\noptimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)\ntf_model.compile(optimizer, "mse", metrics=[rmspe_tf])\n\nvalidation_callback = KerasSequenceValidater(valid_dataset)\nstart = time()\nhistory = tf_model.fit(\n    train_dataset,\n    callbacks=[validation_callback],\n    epochs=EPOCHS,\n)\nt_final = time() - start\ntotal_rows = train_dataset.num_rows_processed + valid_dataset.num_rows_processed\nprint(\n    f"run_time: {t_final} - rows: {total_rows} - epochs: {EPOCHS} - dl_thru: { (EPOCHS * total_rows) / t_final}"\n)\n')


# In[15]:


from nvtabular.inference.triton import export_tensorflow_ensemble
import nvtabular

BASE_DIR = os.environ.get("BASE_DIR", os.path.expanduser("~/nvt-examples/"))
MODEL_NAME_ENSEMBLE = os.environ.get("MODEL_NAME_ENSEMBLE", "rossmann")
# model path to save the models
MODEL_PATH = os.path.join(BASE_DIR, "models/")

workflow = nvtabular.Workflow.load(os.path.join(DATA_DIR, "workflow"))
export_tensorflow_ensemble(tf_model, workflow, MODEL_NAME_ENSEMBLE, MODEL_PATH, LABEL_COLUMNS)


# In[ ]:




