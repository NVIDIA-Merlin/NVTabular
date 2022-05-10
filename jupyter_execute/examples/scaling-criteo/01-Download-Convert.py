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
# # Scaling Criteo: Download and Convert
# 
# ## Criteo 1TB Click Logs dataset
# 
# The [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) is the largest public available dataset for recommender system. It contains ~1.3 TB of uncompressed click logs containing over four billion samples spanning 24 days. Each record contains 40 features: one label indicating a click or no click, 13 numerical figures, and 26 categorical features. The dataset is provided by CriteoLabs. A subset of 7 days was used in this [Kaggle Competition](https://www.kaggle.com/c/criteo-display-ad-challenge/overview). We will use the dataset as an example how to scale ETL, Training and Inference.

# First, we will download the data and extract it. We define the base directory for the dataset and the numbers of day. Criteo provides 24 days. We will use the last day as validation dataset and the remaining days as training. 
# 
# **Each day has a size of ~15GB compressed `.gz` and uncompressed ~XXXGB. You can define a smaller subset of days, if you like. Each day takes ~20-30min to download and extract it.**

# In[2]:


import os

from merlin.core.utils import download_file

download_criteo = True
BASE_DIR = os.environ.get("BASE_DIR", "/raid/data/criteo")
input_path = os.path.join(BASE_DIR, "crit_orig")
NUMBER_DAYS = os.environ.get("NUMBER_DAYS", 2)


# We create the folder structure and download and extract the files. If the file already exist, it will be skipped.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'if download_criteo:\n\n    # Test if NUMBER_DAYS in valid range\n    if NUMBER_DAYS < 2 or NUMBER_DAYS > 23:\n        raise ValueError(\n            str(NUMBER_DAYS)\n            + " is not supported. A minimum of 2 days are "\n            + "required and a maximum of 24 (0-23 days) are available"\n        )\n\n    # Create BASE_DIR if not exists\n    if not os.path.exists(BASE_DIR):\n        os.makedirs(BASE_DIR)\n\n    # Create input dir if not exists\n    if not os.path.exists(input_path):\n        os.makedirs(input_path)\n\n    # Iterate over days\n    for i in range(0, NUMBER_DAYS):\n        file = os.path.join(input_path, "day_" + str(i) + ".gz")\n        # Download file, if there is no .gz, .csv or .parquet file\n        if not (\n            os.path.exists(file)\n            or os.path.exists(\n                file.replace(".gz", ".parquet").replace("crit_orig", "converted/criteo/")\n            )\n            or os.path.exists(file.replace(".gz", ""))\n        ):\n            download_file(\n                "http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_"\n                + str(i)\n                + ".gz",\n                file,\n            )\n')


# The original dataset is in text format. We will convert the dataset into `.parquet` format. Parquet is a compressed, column-oriented file structure and requires less disk space.

# ### Conversion Script for Criteo Dataset (CSV-to-Parquet) 
# 
# __Step 1__: Import libraries

# In[7]:


import os
import glob

import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular as nvt
from merlin.core.utils import device_mem_size, get_rmm_size


# __Step 2__: Specify options
# 
# Specify the input and output paths, unless the `INPUT_DATA_DIR` and `OUTPUT_DATA_DIR` environment variables are already set. For multi-GPU systems, check that the `CUDA_VISIBLE_DEVICES` environment variable includes all desired device IDs.

# In[8]:


INPUT_PATH = os.environ.get("INPUT_DATA_DIR", input_path)
OUTPUT_PATH = os.environ.get("OUTPUT_DATA_DIR", os.path.join(BASE_DIR, "converted"))
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
frac_size = 0.10


# __Step 3__: (Optionally) Start a Dask cluster

# In[9]:


cluster = None  # Connect to existing cluster if desired
if cluster is None:
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
        rmm_pool_size=get_rmm_size(0.8 * device_mem_size()),
        local_directory=os.path.join(OUTPUT_PATH, "dask-space"),
    )
client = Client(cluster)


# __Step 5__: Convert original data to an NVTabular Dataset

# In[10]:


# Specify column names
cont_names = ["I" + str(x) for x in range(1, 14)]
cat_names = ["C" + str(x) for x in range(1, 27)]
cols = ["label"] + cont_names + cat_names

# Specify column dtypes. Note that "hex" means that
# the values will be hexadecimal strings that should
# be converted to int32
dtypes = {}
dtypes["label"] = np.int32
for x in cont_names:
    dtypes[x] = np.int32
for x in cat_names:
    dtypes[x] = "hex"

# Create an NVTabular Dataset from a CSV-file glob
file_list = glob.glob(os.path.join(INPUT_PATH, "day_*[!.gz]"))
dataset = nvt.Dataset(
    file_list,
    engine="csv",
    names=cols,
    part_mem_fraction=frac_size,
    sep="\t",
    dtypes=dtypes,
    client=client,
)


# **__Step 6__**: Write Dataset to Parquet

# In[11]:


dataset.to_parquet(
    os.path.join(OUTPUT_PATH, "criteo"),
    preserve_files=True,
)


# You can delete the original criteo files as they require a lot of disk space.
