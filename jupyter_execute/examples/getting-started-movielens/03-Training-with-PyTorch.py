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
# # Getting Started MovieLens: Training with PyTorch
# 
# ## Overview
# 
# We observed that PyTorch training pipelines can be slow as the dataloader is a bottleneck. The native dataloader in PyTorch randomly sample each item from the dataset, which is very slow. In our experiments, we are able to speed-up existing PyTorch pipelines using a highly optimized dataloader.<br><br>
# 
# Applying deep learning models to recommendation systems faces unique challenges in comparison to other domains, such as computer vision and natural language processing. The datasets and common model architectures have unique characteristics, which require custom solutions. Recommendation system datasets have terabytes in size with billion examples but each example is represented by only a few bytes. For example, the [Criteo CTR dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/), the largest publicly available dataset, is 1.3TB with 4 billion examples. The model architectures have normally large embedding tables for the users and items, which do not fit on a single GPU. You can read more in our [blogpost](https://medium.com/nvidia-merlin/why-isnt-your-recommender-system-training-faster-on-gpu-and-what-can-you-do-about-it-6cb44a711ad4).
# 
# ### Learning objectives
# 
# This notebook explains, how to use the NVTabular dataloader to accelerate PyTorch training.
# 
# 1. Use **NVTabular dataloader** with PyTorch
# 2. Leverage **multi-hot encoded input features**
# 
# ### MovieLens25M
# 
# The [MovieLens25M](https://grouplens.org/datasets/movielens/25m/) is a popular dataset for recommender systems and is used in academic publications. The dataset contains 25M movie ratings for 62,000 movies given by 162,000 users. Many projects use only the user/item/rating information of MovieLens, but the original dataset provides metadata for the movies, as well. For example, which genres a movie has. Although we may not improve state-of-the-art results with our neural network architecture, the purpose of this notebook is to explain how to integrate multi-hot categorical features into a neural network.

# ## NVTabular dataloader for PyTorch
# 
# We’ve identified that the dataloader is one bottleneck in deep learning recommender systems when training pipelines with PyTorch. The dataloader cannot prepare the next batch fast enough, so and therefore, the GPU is not utilized. 
# 
# As a result, we developed a highly customized tabular dataloader for accelerating existing pipelines in PyTorch.  NVTabular dataloader’s features are:
# 
# - removing bottleneck of item-by-item dataloading
# - enabling larger than memory dataset by streaming from disk
# - reading data directly into GPU memory and remove CPU-GPU communication
# - preparing batch asynchronously in GPU to avoid CPU-GPU communication
# - supporting commonly used .parquet format for efficient data format
# - easy integration into existing PyTorch pipelines by using similar API than the native one
# 

# In[2]:


# External dependencies
import os
import gc
import glob

import nvtabular as nvt


# We define our base directory, containing the data.

# In[3]:


INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/movielens/data/")
)


# ### Defining Hyperparameters

# First, we define the data schema and differentiate between single-hot and multi-hot categorical features. Note, that we do not have any numerical input features. 

# In[4]:


BATCH_SIZE = 1024 * 32  # Batch Size
CATEGORICAL_COLUMNS = ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = ["genres"]  # Multi-hot
NUMERIC_COLUMNS = []

# Output from ETL-with-NVTabular
TRAIN_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "train", "*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "valid", "*.parquet")))


# In the previous notebook, we used NVTabular for ETL and stored the workflow to disk. We can load the NVTabular workflow to extract important metadata for our training pipeline.

# In[5]:


proc = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow"))


# The embedding table shows the cardinality of each categorical variable along with its associated embedding size. Each entry is of the form `(cardinality, embedding_size)`.

# In[6]:


EMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)
EMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES


# ### Initializing NVTabular Dataloader for PyTorch

# We import PyTorch and the NVTabular dataloader for PyTorch.

# In[7]:


import torch
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch


# First, we take a look on our dataloader and how the data is represented as tensors. The NVTabular dataloader are initialized as usually and we specify both single-hot and multi-hot categorical features as cats. The dataloader will automatically recognize the single/multi-hot columns and represent them accordingly.

# In[8]:


# TensorItrDataset returns a single batch of x_cat, x_cont, y.

train_dataset = TorchAsyncItr(
    nvt.Dataset(TRAIN_PATHS),
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    conts=NUMERIC_COLUMNS,
    labels=["rating"],
)
train_loader = DLDataLoader(
    train_dataset, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0
)

valid_dataset = TorchAsyncItr(
    nvt.Dataset(VALID_PATHS),
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    conts=NUMERIC_COLUMNS,
    labels=["rating"],
)
valid_loader = DLDataLoader(
    valid_dataset, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0
)


# Let's generate a batch and take a look on the input features.<br><br>
# The single-hot categorical features (`userId` and `movieId`) have a shape of `(32768, 1)`, which is the batch size (as usually). For the multi-hot categorical feature `genres`, we receive two Tensors `genres__values` and `genres__nnzs`.<br><br>
# - `values` are the actual data, containing the genre IDs. Note that the Tensor has more values than the batch_size. The reason is, that one datapoint in the batch can contain more than one genre (multi-hot).<br>
# - `nnzs` are a supporting Tensor, describing how many genres are associated with each datapoint in the batch.<br><br>
# For example,
# - if the first two values in `nnzs` is `0`, `2`, then the first 2 values (0, 1) in `values` are associated with the first datapoint in the batch (movieId/userId).<br>
# - if the next value in `nnzs` is `6`, then the 3rd, 4th and 5th value in `values` are associated with the second datapoint in the batch (continuing after the previous value stopped).<br> 
# - if the third value in `nnzs` is `7`, then the 6th value in `values` are associated with the third datapoint in the batch. 
# - and so on

# In[9]:


batch = next(iter(train_loader))
batch


# `X_cat_multihot` is a tuple of two Tensors. For the multi-hot categorical feature `genres`, we receive two Tensors `values` and `nnzs`.

# In[10]:


X_cat_multihot = batch[0]['genres']
X_cat_multihot


# In[11]:


X_cat_multihot[0].shape


# In[12]:


X_cat_multihot[1].shape


# As each datapoint can have a different number of genres, it is more efficient to represent the genres as two flat tensors: One with the actual values (`values`) and one with the length for each datapoint (`nnzs`).

# In[13]:


del batch
gc.collect()


# ### Defining Neural Network Architecture

# We implemented a simple PyTorch architecture.
# 
# * Single-hot categorical features are fed into an Embedding Layer
# * Each value of a multi-hot categorical features is fed into an Embedding Layer and the multiple Embedding outputs are combined via summing
# * The output of the Embedding Layers are concatenated
# * The concatenated layers are fed through multiple feed-forward layers (Dense Layers, BatchNorm with ReLU activations)
# 
# You can see more details by checking out the implementation.

# In[14]:


# ??Model


# We initialize the model. `EMBEDDING_TABLE_SHAPES` needs to be a Tuple representing the cardinality for single-hot and multi-hot input features.

# In[15]:


EMBEDDING_TABLE_SHAPES_TUPLE = (
    {
        CATEGORICAL_COLUMNS[0]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_COLUMNS[0]],
        CATEGORICAL_COLUMNS[1]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_COLUMNS[1]],
    },
    {CATEGORICAL_MH_COLUMNS[0]: MH_EMBEDDING_TABLE_SHAPES[CATEGORICAL_MH_COLUMNS[0]]},
)
EMBEDDING_TABLE_SHAPES_TUPLE


# In[16]:


model = Model(
    embedding_table_shapes=EMBEDDING_TABLE_SHAPES_TUPLE,
    num_continuous=0,
    emb_dropout=0.0,
    layer_hidden_dims=[128, 128, 128],
    layer_dropout_rates=[0.0, 0.0, 0.0],
).to("cuda")
model


# We initialize the optimizer.

# In[17]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# We use the `process_epoch` function to train and validate our model. It iterates over the dataset and calculates as usually the loss and optimizer step.

# In[18]:


get_ipython().run_cell_magic('time', '', 'from time import time\nEPOCHS = 1\nfor epoch in range(EPOCHS):\n    start = time()\n    train_loss, y_pred, y = process_epoch(train_loader,\n                                          model,\n                                          train=True,\n                                          optimizer=optimizer)\n    valid_loss, y_pred, y = process_epoch(valid_loader,\n                                          model,\n                                          train=False)\n    print(f"Epoch {epoch:02d}. Train loss: {train_loss:.4f}. Valid loss: {valid_loss:.4f}.")\nt_final = time() - start\ntotal_rows = train_dataset.num_rows_processed + valid_dataset.num_rows_processed\nprint(\n    f"run_time: {t_final} - rows: {total_rows * EPOCHS} - epochs: {EPOCHS} - dl_thru: {(total_rows * EPOCHS) / t_final}"\n)\n')

