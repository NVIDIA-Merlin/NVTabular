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
# # NVTabular demo on Rossmann data - PyTorch
# 
# ## Overview
# 
# NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems.  It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.

# ### Learning objectives
# 
# In the previous notebooks ([01-Download-Convert.ipynb](https://github.com/NVIDIA/NVTabular/blob/main/examples/99-applying-to-other-tabular-data-problems-rossmann/01-Download-Convert.ipynb) and [02-ETL-with-NVTabular.ipynb](https://github.com/NVIDIA/NVTabular/blob/main/examples/99-applying-to-other-tabular-data-problems-rossmann/02-ETL-with-NVTabular.ipynb)), we downloaded, preprocessed and created features for the dataset. Now, we are ready to train our deep learning model on the dataset. In this notebook, we use **PyTorch** with the NVTabular data loader for PyTorch to accelereate the training pipeline.

# In[2]:


import os
import math
import json
import nvtabular as nvt
import glob


# ## Loading NVTabular workflow
# This time, we only need to define our data directories. We can load the data schema from the NVTabular workflow.

# In[3]:


DATA_DIR = os.environ.get("OUTPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/data/"))
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


# ## PyTorch<a id="PyTorch"></a>
# 

# ### PyTorch: Preparing Datasets

# In[11]:


import torch
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch

# TensorItrDataset returns a single batch of x_cat, x_cont, y.

train_dataset = TorchAsyncItr(
    nvt.Dataset(TRAIN_PATHS),
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS,
    conts=CONTINUOUS_COLUMNS,
    labels=LABEL_COLUMNS,
)
train_loader = DLDataLoader(
    train_dataset, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0
)

valid_dataset = TorchAsyncItr(
    nvt.Dataset(VALID_PATHS),
    batch_size=BATCH_SIZE,
    cats=CATEGORICAL_COLUMNS,
    conts=CONTINUOUS_COLUMNS,
    labels=LABEL_COLUMNS,
)
valid_loader = DLDataLoader(
    valid_dataset, batch_size=None, collate_fn=lambda x: x, pin_memory=False, num_workers=0
)


# ### PyTorch: Defining a Model

# In[12]:


model = Model(
    embedding_table_shapes=EMBEDDING_TABLE_SHAPES,
    num_continuous=len(CONTINUOUS_COLUMNS),
    emb_dropout=EMBEDDING_DROPOUT_RATE,
    layer_hidden_dims=HIDDEN_DIMS,
    layer_dropout_rates=DROPOUT_RATES,
    max_output=MAX_LOG_SALES_PREDICTION,
).to("cuda")


# ### PyTorch: Training

# In[13]:


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# In[14]:


def rmspe_func(y_pred, y):
    "Return y_pred and y to non-log space and compute RMSPE"
    y_pred, y = torch.exp(y_pred) - 1, torch.exp(y) - 1
    pct_var = (y_pred - y) / y
    return (pct_var ** 2).mean().pow(0.5)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from time import time\n\nstart = time()\nfor epoch in range(EPOCHS):\n    train_loss, y_pred, y = process_epoch(train_loader, model, train=True, optimizer=optimizer)\n    train_rmspe = rmspe_func(y_pred, y)\n    valid_loss, y_pred, y = process_epoch(valid_loader, model, train=False)\n    valid_rmspe = rmspe_func(y_pred, y)\n    print(\n        f"Epoch {epoch:02d}. Train loss: {train_loss:.4f}. Train RMSPE: {train_rmspe:.4f}. Valid loss: {valid_loss:.4f}. Valid RMSPE: {valid_rmspe:.4f}."\n    )\nt_final = time() - start\ntotal_rows = train_dataset.num_rows_processed + valid_dataset.num_rows_processed\nprint(\n    f"run_time: {t_final} - rows: {total_rows} - epochs: {EPOCHS} - dl_thru: {total_rows / t_final}"\n)\n')

