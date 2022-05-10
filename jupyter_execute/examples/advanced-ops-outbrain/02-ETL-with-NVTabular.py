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


# # Getting Started Outbrain: ETL with NVTabular

# ## Overview

# In this notebook we will do preprocessing and feature engineering using [Kaggle Outbrain dataset](https://www.kaggle.com/c/outbrain-click-prediction).
# 
# **Learning objectives**
# 
# In this notebook, we learn how to 
# 
# - Use LambdaOp for custom row-wise dataframe manipulations with NVTabular
# - Preprocess single-hot categorical input features with NVTabular
# - Apply TargetEncoding to categorical features
# - Create a custom operator to create time features
# - Apply ColumnSimilarity to calculate the similarity between two columns using tf-idf metric

# In[2]:


import os
import glob

import cupy

# Get dataframe library - cudf or pandas
from merlin.core.dispatch import get_lib
df_lib = get_lib()

import nvtabular as nvt
from merlin.io import Shuffle
from nvtabular.ops import (
    FillMedian,
    Categorify,
    LogOp,
    TargetEncoding,
    Rename,
)
from nvtabular.ops.column_similarity import ColumnSimilarity

from nvtabular import ColumnGroup


# First, we set where the dataset should be saved once processed (OUTPUT_BUCKET_FOLDER), as well as where the dataset originally resides (DATA_BUCKET_FOLDER).

# In[3]:


DATA_BUCKET_FOLDER = os.environ.get("INPUT_DATA_DIR", "~/nvt-examples/outbrain/data")
OUTPUT_BUCKET_FOLDER = os.environ.get("OUTPUT_DATA_DIR", "./outbrain-preprocessed/")


# Let's read our saved train and valid datasets.

# In[4]:


train_filename = os.path.join(OUTPUT_BUCKET_FOLDER, "train_gdf.parquet")
valid_filename = os.path.join(OUTPUT_BUCKET_FOLDER, "valid_gdf.parquet")


# ## Preparing documents metadata

# Let's create the output directories to store the preprocessed parquet files.

# In[5]:


output_train_dir = os.path.join(OUTPUT_BUCKET_FOLDER, "train/")
output_valid_dir = os.path.join(OUTPUT_BUCKET_FOLDER, "valid/")
get_ipython().system(' mkdir -p $output_train_dir')
get_ipython().system(' mkdir -p $output_valid_dir')


# We read in three more cudf data frames, <i>documents categories</i>, <i>topics</i>, and <i>entities</i>, and use them to create sparse matrices in cupy. We will use these later to calculate cosine similarity between event document (landing page context) and ad document profile vectors (TF-IDF), i.e., how close in profile an ad is to the page that it is being displayed.

# In[6]:


# Alias for read_csv
read_csv = df_lib.read_csv

documents_categories_cudf = read_csv(DATA_BUCKET_FOLDER + "documents_categories.csv")
documents_topics_cudf = read_csv(DATA_BUCKET_FOLDER + "documents_topics.csv")
documents_entities_cudf = read_csv(DATA_BUCKET_FOLDER + "documents_entities.csv")


# read in document categories/topics/entities as cupy sparse matrices
def df_to_coo(df, row="document_id", col=None, data="confidence_level"):
    return cupy.sparse.coo_matrix((df[data].values, (df[row].values, df[col].values)))


categories = df_to_coo(documents_categories_cudf, col="category_id")
topics = df_to_coo(documents_topics_cudf, col="topic_id")
documents_entities_cudf["entity_id"] = (
    documents_entities_cudf["entity_id"].astype("category").cat.codes
)
entities = df_to_coo(documents_entities_cudf, col="entity_id")

documents_categories_cudf = documents_topics_cudf = documents_entities_cudf = None


# ## Initiate NVTabular Workflow

# Now that our datasets, sparse matrices and udf are created, we can begin laying the groundwork for NVTabular. NVTabular requires input features to be defined as groups of columns , so we define our ColumnGroup features at this step. Note that feature engineering and preprocessing often happens to sets of columns, so we adopt that method and require the user to specify continuous and categoricals along with the target as lists within ColumnGroup.

# At this point, our data still isn’t in a form that’s ideal for consumption by our W&D model that we will train in the next notebook. There are missing values, and our categorical variables are still represented by random, discrete identifiers, and need to be transformed into contiguous indices for embedding lookups. The distributions of our continuous variables are uncentered. We also would like to create new features that will help to increase the model accuracy.

# Let's begin to create and process features using NVTabular ops:
#  * <i>geo_location_state</i> and <i>geo_location_country</i> are created by stripping geo_location using the `LambdaOp`
#  * <i>publish_time_days_since_published</i> and <i>publish_time_promo_days_since_published</i> features are created using the `calculate_delta` function in a `LambdaOp`
#  * Missing values are filled using median value depending on the feature using `FillMedian()`op
#  * Continuous features are log transformed with the `LogOp()`.
#  
# `Categorify` op is used for categorification, i.e. encoding of categorical features. Categorify op takes a param called `freq_threshold` which is used for frequency capping. This handy functionality will map all categories which occur in the dataset with some threshold level of infrequency to the _same_ index, keeping the model from overfitting to sparse signals. We don't apply  frequency thresholds in this example, but one can easily create a frequency threshold dictionary, assign a custom threshold value for each categorical feature, and feed that dictionary into the `Categorify` op as `freq_threshold` param.

# One of the important part of building recommender systems is to do feature engineering. As a very promising feature engineering technique, `Target Encoding` processes the categorical features and makes them easier accessible to the model during training and validation. *Target Encoding (TE)* has emerged as being both effective and efficient in many data science projects. For example, it is the major component of Nvidia Kaggle Grandmasters team’s [winning solution](https://medium.com/rapids-ai/winning-solution-of-recsys2020-challenge-gpu-accelerated-feature-engineering-and-training-for-cd67c5a87b1f) of [Recsys Challenge 2020](http://www.recsyschallenge.com/2020/). TE calculates the statistics from a target variable grouped by the unique values of one or more categorical features. For example in a binary classification problem, it calculates the probability that the target is true for each category value - a simple mean. In other words, for each distinct element in feature <b>$x$</b> we are going to compute the average of the corresponding values in target <i>y</i>. Then we are going to replace each $x_{i}$ with the corresponding mean value. For more details on TargetEncoding please visit [here](https://medium.com/rapids-ai/target-encoding-with-rapids-cuml-do-more-with-your-categorical-data-8c762c79e784) and [here](https://github.com/rapidsai/deeplearning/blob/main/RecSys2020Tutorial/03_3_TargetEncoding.ipynb).
# 
# Here, we apply Target Encoding to certain categorical features with *kfold* of 5 and *smoothing* of 20 to avoid overfitting using [TargetEncoding op](https://github.com/NVIDIA/NVTabular/blob/a0141d0a710698470160bc2cbc42b18ce2d49133/nvtabular/ops/target_encoding.py).

# ## Feature Engineering

# Below, we create a custom operator that calculates the time difference between a specified time column (either publish_time or publish_time_promo) and timestamp. This is used to calculate <i>time elapsed since publication</i> between the landing page and the ad.

# In[7]:


# To save disk space, the timestamps in the entire dataset are relative to the first time in the dataset.
# To recover the actual epoch time of the visit, we add 1465876799998 to the timestamp.
TIMESTAMP_DELTA = 1465876799998

from nvtabular.ops import Operator


class DaysSincePublished(Operator):
    def transform(self, columns, gdf):
        for column in columns.names:
            col = gdf[column]
            col.loc[col == ""] = None
            col = col.astype("datetime64[ns]")
            timestamp = (gdf["timestamp"] + TIMESTAMP_DELTA).astype("datetime64[ms]")
            delta = (timestamp - col).dt.days
            gdf[column + "_since_published"] = delta * (delta >= 0) * (delta <= 10 * 365)
        return gdf

    def output_column_names(self, columns):
        return nvt.ColumnSelector([column + "_since_published" for column in columns.names])

    @property
    def dependencies(self):
        return ["timestamp"]


# In[8]:


# geo processing: apply two different lambda operators to the ‘geo_location’ column, and
# extract the country/state from the geo_location value. The geo_location column
# looks something like "US>CA>12345", so we're using string slicing to pull out the country
# and the country+state then
geo_location = ColumnGroup(["geo_location"])
country = geo_location >> (lambda col: col.str.slice(0, 2)) >> Rename(postfix="_country")
state = geo_location >> (lambda col: col.str.slice(0, 5)) >> Rename(postfix="_state")
geo_features = geo_location + country + state

# categoricals processing: categorify certain input columns as well as the geo features
cats = ColumnGroup(
    [
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
)
cat_features = geo_features + cats >> Categorify()

# Apply TargetEncoding to certain categoricals with kfold of 5 and smoothing of 20
te_features = cats >> TargetEncoding("clicked", kfold=5, p_smooth=20)

# process dates using the ‘DaysSincePublished’ custom operator
dates = ["publish_time", "publish_time_promo"]
date_features = dates >> DaysSincePublished() >> FillMedian() >> LogOp()


# Let's visualize our calculation graph with the column groups we used and created so far.

# In[9]:


features = date_features + cat_features + te_features + "clicked"
features.graph


# A user might sometimes be interested to continue reading about the same topics of the current page. Computing the similarity between the textual content of the current page and the pages linked to the displayed ads, can be a relevant feature for a model that predicts which ad the user would click next. A simple, yet effective way to compute the similarity between documents is generating the TF-IDF vectors for each of them, which captures their most relevant terms, and then computing the cosine similarity between those vectors.
#  
# Below, we calculate <i>doc_event_doc_ad_sim_categories</i>, <i>topics</i>, and <i>entities</i> using the `ColumnSimilarity` op, which utilizes the sparse categories, topics, and entities matrices that were created above to calculate landing page similarity for categories, topics, and entities. We calculate Cosine similarity between event doc (landing page) and ad doc aspects vectors (TF-IDF). Creating these extra features help to improve model accuracy and predictability. 

# Note that we rename the column names to avoid duplicated column names.

# In[10]:


sim_features_categ = (
    [["document_id", "document_id_promo"]]
    >> ColumnSimilarity(categories, metric="tfidf", on_device=False)
    >> Rename(postfix="_categories")
)
sim_features_topics = (
    [["document_id", "document_id_promo"]]
    >> ColumnSimilarity(topics, metric="tfidf", on_device=False)
    >> Rename(postfix="_topics")
)
sim_features_entities = (
    [["document_id", "document_id_promo"]]
    >> ColumnSimilarity(entities, metric="tfidf", on_device=False)
    >> Rename(postfix="_entities")
)
sim_features = sim_features_categ + sim_features_topics + sim_features_entities


# In[11]:


# The workflow is created with the output node of the graph
workflow = nvt.Workflow(features + sim_features)


# We then create an NVTabular Dataset object both for train and validation sets. We calculate statistics for this workflow on the input dataset, i.e. on our training set, using the `workflow.fit()` method so that our <i>Workflow</i> can use these stats to transform any given input. When our <i>Workflow</i> transforms our datasets and, we also save the results out to parquet files for fast reading at train time.

# In[12]:


train_dataset = nvt.Dataset(train_filename)
valid_dataset = nvt.Dataset(valid_filename)

# Calculate statistics on the training set
workflow.fit(train_dataset)


# In[ ]:


# use the calculated statistics to transform the train/valid datasets
# and write out each as parquet
workflow.transform(train_dataset).to_parquet(
    output_path=output_train_dir, shuffle=Shuffle.PER_PARTITION, out_files_per_proc=5
)
workflow.transform(valid_dataset).to_parquet(output_path=output_valid_dir)


# We can save the stats from the workflow and load it anytime, so we can run training without doing preprocessing.

# In the next notebooks, we will train a deep learning model. Our training pipeline requires information about the data schema to define the neural network architecture. We will save the NVTabular workflow to disk so that we can restore it in the next notebooks.

# In[ ]:


workflow.save(os.path.join(OUTPUT_BUCKET_FOLDER, "workflow"))


# ## Reviewing processed data

# In[ ]:


TRAIN_PATHS = sorted(glob.glob(os.path.join(OUTPUT_BUCKET_FOLDER, "train/*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(OUTPUT_BUCKET_FOLDER, "valid/*.parquet")))
TRAIN_PATHS, VALID_PATHS


# In[ ]:


df = df_lib.read_parquet(TRAIN_PATHS[0])
df.head()

