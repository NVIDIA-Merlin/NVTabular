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
# ## Overview
# 
# In this notebook, we will show how we do inference with our trained deep learning recommender model using Triton Inference Server. In this example, we deploy the NVTabular workflow and HugeCTR model with Triton Inference Server. We deploy them as an ensemble. For each request, Triton Inference Server will feed the input data through the NVTabular workflow and its output through the HugeCR model.

# As we went through in the previous notebook, [movielens-HugeCTR](https://github.com/NVIDIA/NVTabular/blob/main/examples/inference_triton/inference-HugeCTR/movielens-HugeCTR.ipynb), NVTabular provides a function to save the NVTabular workflow via `export_hugectr_ensemble`. This function does not only save NVTabular workflow, but also saves the trained HugeCTR model and ensemble model to be served to Triton IS.

# ## Getting Started

# We need to write a configuration file with the stored model weights and model configuration.

# In[2]:


get_ipython().run_cell_magic('writefile', "'/model/models/ps.json'", '{\n    "supportlonglong": true,\n    "models": [\n        {\n            "model": "movielens",\n            "sparse_files": ["/model/models/movielens/1/0_sparse_1900.model"],\n            "dense_file": "/model/models/movielens/1/_dense_1900.model",\n            "network_file": "/model/models/movielens/1/movielens.json"\n        }\n    ]\n}\n')


# Let's import required libraries.

# In[3]:


import tritonclient.grpc as httpclient

import numpy as np

# Get dataframe library - cudf or pandas
from nvtabular.dispatch import get_lib
df_lib = get_lib()


# ### Load Models on Triton Server

# At this stage, you should launch the Triton Inference Server docker container with the following script:

# ```
# docker run -it --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/model nvcr.io/nvidia/merlin/merlin-inference:21.11
# ```

# After you started the container you can start triton server with the command below:

# ```
# tritonserver --model-repository=<path_to_models> --backend-config=hugectr,ps=<path_to_models>/ps.json --model-control-mode=explicit
# ```

# Note: The model-repository path is `/model/models/`. The models haven't been loaded, yet. We can request triton server to load the saved ensemble.  We initialize a triton client. The path for the json file is `/model/models/movielens/1/movielens.json`.

# In[4]:


# disable warnings
import warnings

warnings.filterwarnings("ignore")


# In[5]:


import tritonhttpclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))


# In[6]:


triton_client.is_server_live()


# In[7]:


triton_client.get_model_repository_index()


# Let's load our models to Triton Server.

# In[8]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="movielens_nvt")\n')


# In[9]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="movielens")\n')


# Finally, we load our ensemble model `movielens_ens`.

# In[10]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="movielens_ens")\n')


# Let's send a request to Inference Server and print out the response. Since in our example above we do not have continuous columns, below our only inputs are categorical columns.

# In[11]:


from tritonclient.utils import np_to_triton_dtype

model_name = "movielens_ens"
col_names = ["userId", "movieId"]
# read in a batch of data to get transforms for
batch = df_lib.read_parquet("/model/data/valid.parquet", num_rows=64)[col_names]
print(batch, "\n")

# convert the batch to a triton inputs
columns = [(col, batch[col]) for col in col_names]
inputs = []

col_dtypes = [np.int64, np.int64]
for i, (name, col) in enumerate(columns):
    d = col.values_host.astype(col_dtypes[i])
    d = d.reshape(len(d), 1)
    inputs.append(httpclient.InferInput(name, d.shape, np_to_triton_dtype(col_dtypes[i])))
    inputs[i].set_data_from_numpy(d)
# placeholder variables for the output
outputs = []
outputs.append(httpclient.InferRequestedOutput("OUTPUT0"))
# make the request
with httpclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

print("predicted sigmoid result:\n", response.as_numpy("OUTPUT0"))

