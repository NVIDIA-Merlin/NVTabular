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
# # NVTabular demo on Rossmann data - FastAI
# 
# ## Overview
# 
# NVTabular is a feature engineering and preprocessing library for tabular data designed to quickly and easily manipulate terabyte scale datasets used to train deep learning based recommender systems.  It provides a high level abstraction to simplify code and accelerates computation on the GPU using the RAPIDS cuDF library.

# ### Learning objectives
# 
# In the previous notebooks ([01-Download-Convert.ipynb](https://github.com/NVIDIA/NVTabular/blob/main/examples/99-applying-to-other-tabular-data-problems-rossmann/01-Download-Convert.ipynb) and [02-ETL-with-NVTabular.ipynb](https://github.com/NVIDIA/NVTabular/blob/main/examples/99-applying-to-other-tabular-data-problems-rossmann/02-ETL-with-NVTabular.ipynb)), we downloaded, preprocessed and created features for the dataset. Now, we are ready to train our deep learning model on the dataset. In this notebook, we use **FastAI** with the NVTabular data loader for PyTorch to accelereate the training pipeline. FastAI uses PyTorch as a backend and we can combine the NVTabular data loader for PyTorch with the FastAI library.

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


# We load the data schema and statistic information from `stats.json`. We created the file in the previous notebook `02-ETL-with-NVTabular`.

# In[7]:


stats = json.load(open(os.path.join(PREPROCESS_DIR, "stats.json"), "r"))


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
# We'll start by setting some universal hyperparameters for our model and optimizer. These settings will be shared across all of the frameworks that we explore below.

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


# ## fast.ai<a id="fast.ai"></a>
# 

# ### fast.ai: Preparing Datasets
# 
# AsyncTensorBatchDatasetItr maps a symbolic dataset object to `cat_features`, `cont_features`, `labels` PyTorch tenosrs by iterating through the dataset and concatenating the results.

# In[11]:


import fastai

fastai.__version__


# In[12]:


import torch
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader
from nvtabular.framework_utils.torch.utils import FastaiTransform
from fastai.tabular.data import TabularDataLoaders
from fastai.tabular.model import TabularModel
from fastai.basics import Learner
from fastai.basics import MSELossFlat
from fastai.callback.progress import ProgressCallback


def make_batched_dataloader(paths, columns, batch_size):
    dataset = nvt.Dataset(paths)
    ds_batch_sets = TorchAsyncItr(
        dataset,
        batch_size=batch_size,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )
    return DLDataLoader(ds_batch_sets,
                        batch_size=None,
                        pin_memory=False,
                        collate_fn=FastaiTransform(ds_batch_sets).transform,
                        num_workers=0)


train_dataset = make_batched_dataloader(TRAIN_PATHS, COLUMNS, BATCH_SIZE)
valid_dataset = make_batched_dataloader(VALID_PATHS, COLUMNS, BATCH_SIZE * 4)


# In[13]:


databunch = TabularDataLoaders(train_dataset, valid_dataset)


# ### fast.ai: Defining a Model
# 
# Next we'll need to define the inputs that will feed our model and build an architecture on top of them. For now, we'll just stick to a simple MLP model.
# 
# Using FastAI's `TabularModel`, we can build an MLP under the hood by defining its high-level characteristics.

# In[14]:


pt_model = TabularModel(
    emb_szs=list(EMBEDDING_TABLE_SHAPES.values()),
    n_cont=len(CONTINUOUS_COLUMNS),
    out_sz=1,
    layers=HIDDEN_DIMS,
    ps=DROPOUT_RATES,
    use_bn=True,
    embed_p=EMBEDDING_DROPOUT_RATE,
    y_range=torch.tensor([0.0, MAX_LOG_SALES_PREDICTION]),
).cuda()


# ### fast.ai: Training

# In[15]:


from fastai.torch_core import flatten_check
from time import time


def exp_rmspe(pred, targ):
    "Exp RMSE between `pred` and `targ`."
    pred, targ = flatten_check(pred, targ)
    pred, targ = torch.exp(pred) - 1, torch.exp(targ) - 1
    pct_var = (targ - pred) / targ
    return torch.sqrt((pct_var ** 2).mean())


loss_func = MSELossFlat()
learner = Learner(
    databunch, pt_model, loss_func=loss_func, metrics=[exp_rmspe], cbs=ProgressCallback()
)
start = time()
learner.fit(EPOCHS, LEARNING_RATE)
t_final = time() - start
total_rows = train_dataset.dataset.num_rows_processed + valid_dataset.dataset.num_rows_processed
print(
    f"run_time: {t_final} - rows: {total_rows} - epochs: {EPOCHS} - dl_thru: { (EPOCHS * total_rows) / t_final}"
)

