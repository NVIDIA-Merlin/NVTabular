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
# # Getting Started MovieLens: Training with TensorFlow
# 
# ## Overview
# 
# We observed that TensorFlow training pipelines can be slow as the dataloader is a bottleneck. The native dataloader in TensorFlow randomly sample each item from the dataset, which is very slow. The window dataloader in TensorFlow is not much faster. In our experiments, we are able to speed-up existing TensorFlow pipelines by 9x using a highly optimized dataloader.<br><br>
# 
# Applying deep learning models to recommendation systems faces unique challenges in comparison to other domains, such as computer vision and natural language processing. The datasets and common model architectures have unique characteristics, which require custom solutions. Recommendation system datasets have terabytes in size with billion examples but each example is represented by only a few bytes. For example, the [Criteo CTR dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/), the largest publicly available dataset, is 1.3TB with 4 billion examples. The model architectures have normally large embedding tables for the users and items, which do not fit on a single GPU. You can read more in our [blogpost](https://medium.com/nvidia-merlin/why-isnt-your-recommender-system-training-faster-on-gpu-and-what-can-you-do-about-it-6cb44a711ad4).
# 
# ### Learning objectives
# This notebook explains, how to use the NVTabular dataloader to accelerate TensorFlow training.
# 
# 1. Use **NVTabular dataloader** with TensorFlow Keras model
# 2. Leverage **multi-hot encoded input features**
# 
# ### MovieLens25M
# 
# The [MovieLens25M](https://grouplens.org/datasets/movielens/25m/) is a popular dataset for recommender systems and is used in academic publications. The dataset contains 25M movie ratings for 62,000 movies given by 162,000 users. Many projects use only the user/item/rating information of MovieLens, but the original dataset provides metadata for the movies, as well. For example, which genres a movie has. Although we may not improve state-of-the-art results with our neural network architecture, the purpose of this notebook is to explain how to integrate multi-hot categorical features into a neural network.

# ## NVTabular dataloader for TensorFlow
# 
# We’ve identified that the dataloader is one bottleneck in deep learning recommender systems when training pipelines with TensorFlow. The dataloader cannot prepare the next batch fast enough and therefore, the GPU is not fully utilized. 
# 
# We developed a highly customized tabular dataloader for accelerating existing pipelines in TensorFlow. In our experiments, we see a speed-up by 9x of the same training workflow with NVTabular dataloader. NVTabular dataloader’s features are:
# 
# - removing bottleneck of item-by-item dataloading
# - enabling larger than memory dataset by streaming from disk
# - reading data directly into GPU memory and remove CPU-GPU communication
# - preparing batch asynchronously in GPU to avoid CPU-GPU communication
# - supporting commonly used .parquet format
# - easy integration into existing TensorFlow pipelines by using similar API - works with tf.keras models
# 
# More information in our [blogpost](https://medium.com/nvidia-merlin/training-deep-learning-based-recommender-systems-9x-faster-with-tensorflow-cc5a2572ea49).

# In[2]:


# External dependencies
import os
import glob

import nvtabular as nvt


# We define our base input directory, containing the data.

# In[3]:


INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/movielens/data/")
)
# path to save the models
MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", os.path.expanduser("~/nvt-examples/"))


# In[4]:


# avoid numba warnings
from numba import config
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


# ### Defining Hyperparameters

# First, we define the data schema and differentiate between single-hot and multi-hot categorical features. Note, that we do not have any numerical input features. 

# In[5]:


BATCH_SIZE = 1024 * 32  # Batch Size
CATEGORICAL_COLUMNS = ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = ["genres"]  # Multi-hot
NUMERIC_COLUMNS = []

# Output from ETL-with-NVTabular
TRAIN_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "train", "*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "valid", "*.parquet")))


# In the previous notebook, we used NVTabular for ETL and stored the workflow to disk. We can load the NVTabular workflow to extract important metadata for our training pipeline.

# In[6]:


workflow = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow"))


# The embedding table shows the cardinality of each categorical variable along with its associated embedding size. Each entry is of the form `(cardinality, embedding_size)`.

# In[7]:


EMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(workflow)
EMBEDDING_TABLE_SHAPES.update(MH_EMBEDDING_TABLE_SHAPES)
EMBEDDING_TABLE_SHAPES


# ### Initializing NVTabular Dataloader for Tensorflow

# We import TensorFlow and some NVTabular TF extensions, such as custom TensorFlow layers supporting multi-hot and the NVTabular TensorFlow data loader.

# In[8]:


import os
import time
import tensorflow as tf

from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.framework_utils.tensorflow import layers


# First, we take a look on our data loader and how the data is represented as tensors. The NVTabular data loader are initialized as usually and we specify both single-hot and multi-hot categorical features as cat_names. The data loader will automatically recognize the single/multi-hot columns and represent them accordingly.

# In[9]:


train_dataset_tf = KerasSequenceLoader(
    TRAIN_PATHS,  # you could also use a glob pattern
    batch_size=BATCH_SIZE,
    label_names=["rating"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.06,  # how many batches to load at once
    parts_per_chunk=1,
)

valid_dataset_tf = KerasSequenceLoader(
    VALID_PATHS,  # you could also use a glob pattern
    batch_size=BATCH_SIZE,
    label_names=["rating"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=False,
    buffer_size=0.06,
    parts_per_chunk=1,
)


# Let's generate a batch and take a look on the input features.<br><br>
# We can see, that the single-hot categorical features (`userId` and `movieId`) have a shape of `(32768, 1)`, which is the batchsize (as usually).<br><br>
# For the multi-hot categorical feature `genres`, we receive two Tensors `genres__values` and `genres__nnzs`.<br><br>
# `genres__values` are the actual data, containing the genre IDs. Note that the Tensor has more values than the batch_size. The reason is, that one datapoint in the batch can contain more than one genre (multi-hot).<br>
# `genres__nnzs` are a supporting Tensor, describing how many genres are associated with each datapoint in the batch.<br><br>
# For example,
# - if the first value in `genres__nnzs` is `5`, then the first 5 values in `genres__values` are associated with the first datapoint in the batch (movieId/userId).<br>
# - if the second value in `genres__nnzs` is `2`, then the 6th and the 7th values in `genres__values` are associated with the second datapoint in the batch (continuing after the previous value stopped).<br> 
# - if the third value in `genres_nnzs` is `1`, then the 8th value in `genres__values` are associated with the third datapoint in the batch. 
# - and so on

# In[10]:


batch = next(iter(train_dataset_tf))
batch[0]


# We can see that the sum of `genres__nnzs` is equal to the shape of `genres__values`.

# In[11]:


tf.reduce_sum(batch[0]["genres"][1])


# As each datapoint can have a different number of genres, it is more efficient to represent the genres as two flat tensors: One with the actual values (`genres__values`) and one with the length for each datapoint (`genres__nnzs`).

# In[12]:


del batch


# ### Defining Neural Network Architecture

# We will define a common neural network architecture for tabular data.
# 
# * Single-hot categorical features are fed into an Embedding Layer
# * Each value of a multi-hot categorical features is fed into an Embedding Layer and the multiple Embedding outputs are combined via averaging
# * The output of the Embedding Layers are concatenated
# * The concatenated layers are fed through multiple feed-forward layers (Dense Layers with ReLU activations)
# * The final output is a single number with sigmoid activation function

# First, we will define some dictionary/lists for our network architecture.

# In[13]:


inputs = {}  # tf.keras.Input placeholders for each feature to be used
emb_layers = []  # output of all embedding layers, which will be concatenated


# We create `tf.keras.Input` tensors for all 4 input features.

# In[14]:


for col in CATEGORICAL_COLUMNS:
    inputs[col] = tf.keras.Input(name=col, dtype=tf.int64, shape=(1,))
# Note that we need two input tensors for multi-hot categorical features
for col in CATEGORICAL_MH_COLUMNS:
    inputs[col] = (tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,)),
                   tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,)))


# Next, we initialize Embedding Layers with `tf.feature_column.embedding_column`.

# In[15]:


for col in CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS:
    emb_layers.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                col, EMBEDDING_TABLE_SHAPES[col][0]
            ),  # Input dimension (vocab size)
            EMBEDDING_TABLE_SHAPES[col][1],  # Embedding output dimension
        )
    )
emb_layers


# NVTabular implemented a custom TensorFlow layer `layers.DenseFeatures`, which takes as an input the different `tf.Keras.Input` and pre-initialized `tf.feature_column` and automatically concatenate them into a flat tensor. In the case of multi-hot categorical features, `DenseFeatures` organizes the inputs `__values` and `__nnzs` to define a `RaggedTensor` and combine them. `DenseFeatures` can handle numeric inputs, as well, but MovieLens does not provide numerical input features.

# In[16]:


emb_layer = layers.DenseFeatures(emb_layers)
x_emb_output = emb_layer(inputs)
x_emb_output


# We can see that the output shape of the concatenated layer is equal to the sum of the individual Embedding output dimensions (1040 = 16+512+512).
# 

# In[17]:


EMBEDDING_TABLE_SHAPES


# We add multiple Dense Layers. Finally, we initialize the `tf.keras.Model` and add the optimizer.

# In[18]:


x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

model = tf.keras.Model(inputs=inputs, outputs=x)
model.compile("sgd", "binary_crossentropy")


# In[19]:


# You need to install the dependencies
tf.keras.utils.plot_model(model)


# ### Training the deep learning model

# We can train our model with `model.fit`. We need to use a Callback to add the validation dataloader.

# In[20]:


validation_callback = KerasSequenceValidater(valid_dataset_tf)
EPOCHS = 1
start = time.time()
history = model.fit(train_dataset_tf, callbacks=[validation_callback], epochs=EPOCHS)
t_final = time.time() - start
total_rows = train_dataset_tf.num_rows_processed + valid_dataset_tf.num_rows_processed
print(
    f"run_time: {t_final} - rows: {total_rows * EPOCHS} - epochs: {EPOCHS} - dl_thru: {(EPOCHS * total_rows) / t_final}"
)


# In[21]:


MODEL_NAME_TF = os.environ.get("MODEL_NAME_TF", "movielens_tf")
MODEL_PATH_TEMP_TF = os.path.join(MODEL_BASE_DIR, MODEL_NAME_TF, "1/model.savedmodel")

model.save(MODEL_PATH_TEMP_TF)


# Before moving to the next notebook, `04a-Triton-Inference-with-TF.ipynb`, we need to generate the Triton Inference Server configurations and save the models in the correct format. We just saved TensorFlow model to disk, and in the previous notebook `02-ETL-with-NVTabular`, we saved the NVTabular workflow. Let's load the workflow. 

# The TensorFlow input layers expect the input datatype to be int32. Therefore, we need to change the output datatypes to int32 for our NVTabular workflow.

# In[22]:


workflow = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow"))


# In[23]:


MODEL_NAME_ENSEMBLE = os.environ.get("MODEL_NAME_ENSEMBLE", "movielens")
# model path to save the models
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(MODEL_BASE_DIR, "models"))


# NVTabular provides a function to save the NVTabular workflow, TensorFlow model and Triton Inference Server (IS) config files via `export_tensorflow_ensemble`. We provide the model, workflow, a model name for ensemble model, path and output column.

# In[24]:


# Creates an ensemble triton server model, where
#  model: The tensorflow model that should be served
#  workflow: The nvtabular workflow used in preprocessing
#  name: The base name of the various triton models

from nvtabular.inference.triton import export_tensorflow_ensemble
export_tensorflow_ensemble(model, workflow, MODEL_NAME_ENSEMBLE, MODEL_PATH, ["rating"])


# Now, we can move to the next notebook, [04-Triton-Inference-with-TF.ipynb](https://github.com/NVIDIA/NVTabular/blob/main/examples/getting-started-movielens/04-Triton-Inference-with-TF.ipynb), to send inference request to the Triton IS.
