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


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Getting Started MovieLens: Download and Convert
# 
# ## MovieLens25M
# 
# The [MovieLens25M](https://grouplens.org/datasets/movielens/25m/) is a popular dataset for recommender systems and is used in academic publications. The dataset contains 25M movie ratings for 62,000 movies given by 162,000 users. Many projects use only the user/item/rating information of MovieLens, but the original dataset provides metadata for the movies, as well. For example, which genres a movie has. Although we may not improve state-of-the-art results with our neural network architecture in this example, we will use the metadata to show how to multi-hot encode the categorical features.

# ## Download the dataset

# In[2]:


# External dependencies
import os

from merlin.core.utils import download_file

# Get dataframe library - cudf or pandas
from merlin.core.dispatch import get_lib
df_lib = get_lib()


# We define our base input directory, containing the data.

# In[3]:


INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/movielens/data/")
)


# We will download and unzip the data.

# In[4]:


download_file(
    "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
    os.path.join(INPUT_DATA_DIR, "ml-25m.zip"),
)


# ## Convert the dataset

# First, we take a look on the movie metadata. 

# In[5]:


movies = df_lib.read_csv(os.path.join(INPUT_DATA_DIR, "ml-25m/movies.csv"))
movies.head()


# We can see, that genres are a multi-hot categorical features with different number of genres per movie. Currently, genres is a String and we want split the String into a list of Strings. In addition, we drop the title.

# In[6]:


movies["genres"] = movies["genres"].str.split("|")
movies = movies.drop("title", axis=1)
movies.head()


# We save movies genres in parquet format, so that they can be used by NVTabular in the next notebook.

# In[7]:


movies.to_parquet(os.path.join(INPUT_DATA_DIR, "movies_converted.parquet"))


# ## Splitting into train and validation dataset

# We load the movie ratings.

# In[8]:


ratings = df_lib.read_csv(os.path.join(INPUT_DATA_DIR, "ml-25m", "ratings.csv"))
ratings.head()


# We drop the timestamp column and split the ratings into training and test datasets. We use a simple random split.

# In[9]:


ratings = ratings.drop("timestamp", axis=1)

# shuffle the dataset
ratings = ratings.sample(len(ratings), replace=False)

# split the train_df as training and validation data sets.
num_valid = int(len(ratings) * 0.2)

train = ratings[:-num_valid]
valid = ratings[-num_valid:]


# We save the dataset to disk.

# In[10]:


train.to_parquet(os.path.join(INPUT_DATA_DIR, "train.parquet"))
valid.to_parquet(os.path.join(INPUT_DATA_DIR, "valid.parquet"))

