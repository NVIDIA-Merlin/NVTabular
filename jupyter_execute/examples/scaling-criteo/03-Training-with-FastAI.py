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
# # Scaling Criteo: Training with FastAI
# 
# ## Overview
# 
# We observed that training pipelines can be slow as the dataloader is a bottleneck. NVTabular provides a highly optimized dataloader to accelerate training pipelines. We can use the PyTorch dataloader for FastAI models.
# 
# We have already discussed the NVTabular dataloaders in more detail in our [Getting Started with Movielens notebooks](https://github.com/NVIDIA/NVTabular/tree/main/examples/getting-started-movielens).<br><br>
# 
# We will use the same techniques to train a deep learning model for the [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).
# 
# ### Learning objectives
# 
# In this notebook, we learn how to:
# 
# - Use **NVTabular dataloader** with FastAI Tabular model

# ## NVTabular dataloader for PyTorch / FastAI
# 
# When training pipelines with PyTorch, the dataloader cannot prepare sequential batches fast enough, so the GPU is not fully utilized. To combat this issue, we’ve developed a highly customized tabular dataloader, `TorchAsyncItr`, to accelerate existing pipelines in PyTorch. The NVTabular dataloader is capable of:
# 
# - removing bottlenecks from dataloading by processing large chunks of data at a time instead of item by item
# - processing datasets that don’t fit within the GPU or CPU memory by streaming from the disk
# - reading data directly into the GPU memory and removing CPU-GPU communication
# - preparing batch asynchronously into the GPU to avoid CPU-GPU communication
# - supporting commonly used formats such as parquet
# - integrating easily into existing PyTorch training pipelines by using a similar API as the native PyTorch dataloader

# In[2]:


import os
from time import time
import glob

# tools for data preproc/loading
import torch
import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
from nvtabular.framework_utils.torch.utils import FastaiTransform

# tools for training
from fastai.basics import Learner
from fastai.tabular.model import TabularModel
from fastai.tabular.data import TabularDataLoaders
from fastai.metrics import RocAucBinary, APScoreBinary


# ### Dataset and Dataset Schema
# Once our data is ready, we'll define some high level parameters to describe where our data is and what it "looks like" at a high level.

# ### Data Loading
# We'll start by using the parquet files we just created to feed an NVTabular `TorchAsyncItr`, which will loop through the files in chunks. First, we'll reinitialize our memory pool from earlier to free up some memory so that we can share it with PyTorch.

# In[3]:


CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
LABEL_COLUMNS = ["label"]
BASE_DIR = os.environ.get("BASE_DIR", "/raid/data/criteo")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 400000))
PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))
input_path = os.environ.get("INPUT_DATA_DIR", os.path.join(BASE_DIR, "test_dask/output"))


# In[4]:


train_paths = glob.glob(os.path.join(input_path, "train", "*.parquet"))
valid_paths = glob.glob(os.path.join(input_path, "valid", "*.parquet"))


# In[5]:


train_data = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)
valid_data = nvt.Dataset(valid_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)


# In[6]:


train_data_itrs = TorchAsyncItr(
    train_data,
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS,
    conts=CONTINUOUS_COLUMNS,
    labels=LABEL_COLUMNS,
    parts_per_chunk=PARTS_PER_CHUNK,
)
valid_data_itrs = TorchAsyncItr(
    valid_data,
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS,
    conts=CONTINUOUS_COLUMNS,
    labels=LABEL_COLUMNS,
    parts_per_chunk=PARTS_PER_CHUNK,
)


# In[7]:


def gen_col(batch):
    return (batch[0], batch[1], batch[2].long())


# In[8]:


train_dataloader = DLDataLoader(
    train_data_itrs,
    collate_fn=FastaiTransform(train_data_itrs).transform,
    batch_size=None,
    pin_memory=False,
    num_workers=0
)
valid_dataloader = DLDataLoader(
    valid_data_itrs,
    collate_fn=FastaiTransform(valid_data_itrs).transform,
    batch_size=None,
    pin_memory=False,
    num_workers=0
)
databunch = TabularDataLoaders(train_dataloader, valid_dataloader)


# Now we have data ready to be fed to our model online!

# ### Training
# One extra handy functionality of NVTabular is the ability to use the stats collected by the `Categorify` op to define embedding dictionary sizes (i.e. the number of rows of your embedding table). It even includes a heuristic for computing a good embedding size (i.e. the number of columns of your embedding table) based off of the number of categories.

# In the previous notebook, we used NVTabular for ETL and stored the workflow to disk. We can load the NVTabular workflow to extract important metadata for our training pipeline.

# In[9]:


workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))


# In[10]:


embeddings = list(get_embedding_sizes(workflow).values())
# We limit the output dimension to 16
embeddings = [[emb[0], min(16, emb[1])] for emb in embeddings]
embeddings


# In[11]:


model = TabularModel(
    emb_szs=embeddings, n_cont=len(CONTINUOUS_COLUMNS), out_sz=2, layers=[512, 256]
).cuda()
learn = Learner(
    databunch,
    model,
    loss_func=torch.nn.CrossEntropyLoss(),
    metrics=[RocAucBinary(), APScoreBinary()],
)


# In[12]:


learning_rate = 1.32e-2
epochs = 1
start = time()
learn.fit(epochs, learning_rate)
t_final = time() - start
total_rows = train_data_itrs.num_rows_processed + valid_data_itrs.num_rows_processed
print(
    f"run_time: {t_final} - rows: {total_rows} - epochs: {epochs} - dl_thru: {total_rows / t_final}"
)

