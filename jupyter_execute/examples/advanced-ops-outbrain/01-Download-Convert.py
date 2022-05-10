#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # Getting Started Outbrain: Download and Convert

# ## Overview

# Outbrain dataset was published in [Kaggle Outbrain click prediction](https://www.kaggle.com/c/outbrain-click-prediction) competition, where the ‘Kagglers’ were challenged to predict on which ads and other forms of sponsored content its global users would click. One of  the top finishers' preprocessing and feature engineering pipeline is taken into consideration here, and this pipeline was restructured using NVTabular and cuDF. 

# In[2]:


import os

# Get dataframe library - cudf or pandas
from merlin.core.dispatch import get_lib, random_uniform, reinitialize
df_lib = get_lib()


# ## Download the dataset

# First, you need to [download](https://www.kaggle.com/c/outbrain-click-prediction/data) the Kaggle Outbrain click prediction challenge and set DATA_BUCKET_FOLDER with the dataset path.  

# In[3]:


DATA_BUCKET_FOLDER = os.environ.get("INPUT_DATA_DIR", "~/nvt-examples/outbrain/data/")


# The OUTPUT_BUCKET_FOLDER is the folder where the preprocessed dataset will be saved.

# In[4]:


OUTPUT_BUCKET_FOLDER = os.environ.get("OUTPUT_DATA_DIR", "./outbrain-preprocessed/")
os.makedirs(OUTPUT_BUCKET_FOLDER, exist_ok=True)


# ## Preparing Our Dataset

# Here, we merge the component tables of our dataset into a single data frame, using [cuDF](https://github.com/rapidsai/cudf), which is a GPU DataFrame library for loading, joining, aggregating, filtering, and otherwise manipulating data. We do this because NVTabular applies a workflow to a single table. We also re-initialize managed memory. `rmm.reinitialize()` provides an easy way to initialize RMM (RAPIDS Memory Manager) with specific memory resource options across multiple devices. The reason we re-initialize managed memory here is to allow us to perform memory intensive merge operation. Note that dask-cudf can also be used here.

# In[5]:


# use managed memory for device memory allocation
reinitialize(managed_memory=True)

# Alias for read_csv
read_csv = df_lib.read_csv

# Merge all the CSV files together
documents_meta = read_csv(DATA_BUCKET_FOLDER + "documents_meta.csv", na_values=["\\N", ""])
merged = (
    read_csv(DATA_BUCKET_FOLDER + "clicks_train.csv", na_values=["\\N", ""])
    .merge(
        read_csv(DATA_BUCKET_FOLDER + "events.csv", na_values=["\\N", ""]),
        on="display_id",
        how="left",
        suffixes=("", "_event"),
    )
    .merge(
        read_csv(DATA_BUCKET_FOLDER + "promoted_content.csv", na_values=["\\N", ""]),
        on="ad_id",
        how="left",
        suffixes=("", "_promo"),
    )
    .merge(documents_meta, on="document_id", how="left")
    .merge(
        documents_meta,
        left_on="document_id_promo",
        right_on="document_id",
        how="left",
        suffixes=("", "_promo"),
    )
)


# ## Splitting into train and validation datasets

# We use a time-stratified sample to create a validation set that is more recent, and save both our train and validation sets to parquet files to be read by NVTabular. Note that you should run the cell below only once, then save your `train` and `valid` data frames as parquet files. If you want to rerun this notebook you might end up with a different train-validation split each time because samples are drawn from a uniform distribution.

# In[6]:


# Do a stratified split of the merged dataset into a training/validation dataset
merged["day_event"] = (merged["timestamp"] / 1000 / 60 / 60 / 24).astype(int)
random_state = df_lib.Series(random_uniform(size=len(merged)))
valid_set, train_set = merged.scatter_by_map(
    ((merged.day_event <= 10) & (random_state > 0.2)).astype(int)
)


# In[7]:


train_set.head()


# We save the dataset to disk.

# In[8]:


train_filename = os.path.join(OUTPUT_BUCKET_FOLDER, "train_gdf.parquet")
valid_filename = os.path.join(OUTPUT_BUCKET_FOLDER, "valid_gdf.parquet")
train_set.to_parquet(train_filename, compression=None)
valid_set.to_parquet(valid_filename, compression=None)
merged = train_set = valid_set = None


# In[9]:


reinitialize(managed_memory=False)

