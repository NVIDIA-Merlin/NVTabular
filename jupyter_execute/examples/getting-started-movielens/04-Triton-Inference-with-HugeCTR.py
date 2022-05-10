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
# ===================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Getting Started MovieLens: Serving a HugeCTR Model
# 
# In this notebook, we will show how we do inference with our trained deep learning recommender model using Triton Inference Server. In this example, we deploy the NVTabular workflow and HugeCTR model with Triton Inference Server. We deploy them as an ensemble. For each request, Triton Inference Server will feed the input data through the NVTabular workflow and its output through the HugeCR model.

# ## Getting Started

# We need to write configuration files with the stored model weights and model configuration.

# In[2]:


get_ipython().run_cell_magic('writefile', '/model/movielens_hugectr/config.pbtxt', 'name: "movielens_hugectr"\nbackend: "hugectr"\nmax_batch_size: 64\ninput [\n   {\n    name: "DES"\n    data_type: TYPE_FP32\n    dims: [ -1 ]\n  },\n  {\n    name: "CATCOLUMN"\n    data_type: TYPE_INT64\n    dims: [ -1 ]\n  },\n  {\n    name: "ROWINDEX"\n    data_type: TYPE_INT32\n    dims: [ -1 ]\n  }\n]\noutput [\n  {\n    name: "OUTPUT0"\n    data_type: TYPE_FP32\n    dims: [ -1 ]\n  }\n]\ninstance_group [\n  {\n    count: 1\n    kind : KIND_GPU\n    gpus:[0]\n  }\n]\n\nparameters [\n  {\n  key: "config"\n  value: { string_value: "/model/movielens_hugectr/1/movielens.json" }\n  },\n  {\n  key: "gpucache"\n  value: { string_value: "true" }\n  },\n  {\n  key: "hit_rate_threshold"\n  value: { string_value: "0.8" }\n  },\n  {\n  key: "gpucacheper"\n  value: { string_value: "0.5" }\n  },\n  {\n  key: "label_dim"\n  value: { string_value: "1" }\n  },\n  {\n  key: "slots"\n  value: { string_value: "3" }\n  },\n  {\n  key: "cat_feature_num"\n  value: { string_value: "4" }\n  },\n {\n  key: "des_feature_num"\n  value: { string_value: "0" }\n  },\n  {\n  key: "max_nnz"\n  value: { string_value: "2" }\n  },\n  {\n  key: "embedding_vector_size"\n  value: { string_value: "16" }\n  },\n  {\n  key: "embeddingkey_long_type"\n  value: { string_value: "true" }\n  }\n]\n')


# In[3]:


get_ipython().run_cell_magic('writefile', '/model/ps.json', '{\n    "supportlonglong":true,\n    "models":[\n        {\n            "model":"movielens_hugectr",\n            "sparse_files":["/model/movielens_hugectr/0_sparse_1900.model"],\n            "dense_file":"/model/movielens_hugectr/_dense_1900.model",\n            "network_file":"/model/movielens_hugectr/1/movielens.json",\n            "num_of_worker_buffer_in_pool": "1",\n            "num_of_refresher_buffer_in_pool": "1",\n            "cache_refresh_percentage_per_iteration": "0.2",\n            "deployed_device_list":["0"],\n            "max_batch_size":"64",\n            "default_value_for_each_table":["0.0","0.0"],\n            "hit_rate_threshold":"0.9",\n            "gpucacheper":"0.5",\n            "gpucache":"true"\n        }\n    ]  \n}\n')


# Let's import required libraries.

# In[4]:


import tritonclient.grpc as httpclient
import cudf
import numpy as np


# ### Load Models on Triton Server

# At this stage, you should launch the Triton Inference Server docker container with the following script:

# ```
# docker run -it --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/model nvcr.io/nvidia/merlin/merlin-inference:21.11
# ```

# After you started the container you can start triton server with the command below:

# ```
# tritonserver --model-repository=<path_to_models> --backend-config=hugectr,ps=<path_to_models>/ps.json --model-control-mode=explicit
# ```

# Note: The model-repository path is `/model/`. The models haven't been loaded, yet. We can request triton server to load the saved ensemble.  We initialize a triton client. The path for the json file is `/model/movielens_hugectr/1/movielens.json`.

# In[5]:


# disable warnings
import warnings

warnings.filterwarnings("ignore")


# In[6]:


import tritonhttpclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))


# In[7]:


triton_client.is_server_live()


# In[8]:


triton_client.get_model_repository_index()


# Let's load our model to Triton Server.

# In[9]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="movielens_hugectr")\n')


# Let's send a request to Inference Server and print out the response. Since in our example above we do not have continuous columns, below our only inputs are categorical columns.

# In[10]:


import pandas as pd
df = pd.read_parquet("/model/data/valid/part_0.parquet")


# In[11]:


df.head()


# In[12]:


get_ipython().run_cell_magic('writefile', './wdl2predict.py', 'from tritonclient.utils import *\nimport tritonclient.http as httpclient\nimport numpy as np\nimport pandas as pd\nimport sys\n\nmodel_name = \'movielens_hugectr\'\nCATEGORICAL_COLUMNS = ["userId", "movieId", "genres"]\nCONTINUOUS_COLUMNS = []\nLABEL_COLUMNS = [\'label\']\nemb_size_array = [162542, 29434, 20]\nshift = np.insert(np.cumsum(emb_size_array), 0, 0)[:-1]\ndf = pd.read_parquet("/model/data/valid/part_0.parquet")\ntest_df = df.head(10)\n\nrp_lst = [0]\ncur = 0\nfor i in range(1, 31):\n    if i % 3 == 0:\n        cur += 2\n        rp_lst.append(cur)\n    else:\n        cur += 1\n        rp_lst.append(cur)\n\nwith httpclient.InferenceServerClient("localhost:8000") as client:\n    test_df.iloc[:, :2] = test_df.iloc[:, :2] + shift[:2]\n    test_df.iloc[:, 2] = test_df.iloc[:, 2].apply(lambda x: [e + shift[2] for e in x])\n    embedding_columns = np.array([list(np.hstack(np.hstack(test_df[CATEGORICAL_COLUMNS].values)))], dtype=\'int64\')\n    dense_features = np.array([[]], dtype=\'float32\')\n    row_ptrs = np.array([rp_lst], dtype=\'int32\')\n\n    inputs = [httpclient.InferInput("DES", dense_features.shape, np_to_triton_dtype(dense_features.dtype)),\n              httpclient.InferInput("CATCOLUMN", embedding_columns.shape, np_to_triton_dtype(embedding_columns.dtype)),\n              httpclient.InferInput("ROWINDEX", row_ptrs.shape, np_to_triton_dtype(row_ptrs.dtype))]\n\n    inputs[0].set_data_from_numpy(dense_features)\n    inputs[1].set_data_from_numpy(embedding_columns)\n    inputs[2].set_data_from_numpy(row_ptrs)\n    outputs = [httpclient.InferRequestedOutput("OUTPUT0")]\n\n    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)\n\n    result = response.get_response()\n    print(result)\n    print("Prediction Result:")\n    print(response.as_numpy("OUTPUT0"))\n')


# In[13]:


get_ipython().system('python3 ./wdl2predict.py')

