#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2020 NVIDIA Corporation. All Rights Reserved.
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
# # Getting Started MovieLens: Serving a TensorFlow Model
# The last step is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as done during training ETL. We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integers before we use the deep learning model for a prediction. Therefore, we deploy the NVTabular workflow with the TensorFlow model as an ensemble model to Triton Inference. The ensemble model guarantees that the same transformation are applied to the raw inputs.

# <center><img src="./imgs/triton-tf.png" width="300" height="200"></center>

# ## Learning Objectives
# In the previous notebook we explained and showed how we can preprocess data with multi-hot columns with NVTabular, and train an TF MLP model using NVTabular `KerasSequenceLoader`. We learned how to save a workflow, a trained TF model, and the ensemble model. In this notebook, we will show an example request script sent to the Triton Inference Server. We will learn
# 
# - to transform new/streaming data with NVTabular library
# - to deploy the end-to-end pipeline to generate prediction results for new data from trained TF model

# ## Starting Triton Inference Server

# Before we get started, start Triton Inference Server in the Docker container with the following command. The command includes the `-v` argument to mount your local `model-repository` directory that includes your saved models from the previous notebook (`03a-Training-with-TF.ipynb`) to `/model` directory in the container.

# ```
# docker run -it --gpus device=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}:/model/ nvcr.io/nvidia/merlin/merlin-tensorflow:latest
# ```

# After you start the container, you can start Triton Inference Server with the following command. You need to provide correct path for the `models` directory.
# 
# ```
# tritonserver --model-repository=path_to_models --backend-config=tensorflow,version=2 --model-control-mode=explicit 
# ```

# Note: The model-repository path is `/model/nvt-examples/models/`. The models haven't been loaded, yet. Below, we will request the Triton server to load the saved ensemble model.

# In[2]:


# External dependencies
import os
from time import time

# Get dataframe library - cudf or pandas
from merlin.core.dispatch import get_lib
df_lib = get_lib()

import tritonclient.grpc as grpcclient
import nvtabular.inference.triton as nvt_triton


# We define our base directory, containing the data.

# In[3]:


# path to preprocessed data
INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/movielens/data/")
)


# Let's deactivate the warnings before sending requests.

# In[4]:


import warnings

warnings.filterwarnings("ignore")


# ## Loading Ensemble Model with Triton Inference Server

# At this stage, you should have started the Triton Inference Server in a container with the instructions above.

# Let's connect to the Triton Inference Server. Use Triton’s ready endpoint to verify that the server and the models are ready for inference. Replace localhost with your host ip address.

# In[5]:


import tritonhttpclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="localhost:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))


# In[6]:


import warnings

warnings.filterwarnings("ignore")


# We check if the server is alive.

# In[7]:


triton_client.is_server_live()


# The HTTP request returns status 200 if Triton is ready and non-200 if it is not ready.

# We check the available models in the repositories:
# 
# movielens: Ensemble <br>
# movielens_nvt: NVTabular <br>
# movielens_tf: TensorFlow model

# In[8]:


triton_client.get_model_repository_index()


# We load the ensemble model.

# In[9]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="movielens")\n')


# ## Send request to Triton Inference Server to transform raw dataset

# A minimal model repository for a TensorFlow SavedModel model is:
# ```
#   <model-repository-path>/<model-name>/
#       config.pbtxt
#       1/
#         model.savedmodel/
#            <saved-model files>
# ```
# Let's check out our model repository layout. You can install tree library with `apt-get install tree`, and then run `!tree /model/models/` to print out the model repository layout as below:
#                
# ```
# /model/models/
# |-- movielens
# |   |-- 1
# |   `-- config.pbtxt
# |-- movielens_nvt
# |   |-- 1
# |   |   |-- __pycache__
# |   |   |   `-- model.cpython-38.pyc
# |   |   |-- model.py
# |   |   `-- workflow
# |   |       |-- categories
# |   |       |   |-- unique.genres.parquet
# |   |       |   |-- unique.movieId.parquet
# |   |       |   `-- unique.userId.parquet
# |   |       |-- metadata.json
# |   |       `-- workflow.pkl
# |   `-- config.pbtxt
# `-- movielens_tf
#     |-- 1
#     |   `-- model.savedmodel
#     |       |-- assets
#     |       |-- saved_model.pb
#     |       `-- variables
#     |           |-- variables.data-00000-of-00001
#     |           `-- variables.index
#     `-- config.pbtxt
# ```
# You can see that we have a `config.pbtxt` file. Each model in a model repository must include a model configuration that provides required and optional information about the model. Typically, this configuration is provided in a `config.pbtxt` file specified as [ModelConfig protobuf](https://github.com/triton-inference-server/server/blob/r20.12/src/core/model_config.proto).

# Let's read the raw validation set, and send 3 rows of `userId` and `movieId` as input to the saved NVTabular model.

# In[10]:


# read in the workflow (to get input/output schema to call triton with)
batch = df_lib.read_parquet(
    os.path.join(INPUT_DATA_DIR, "valid.parquet"), num_rows=3, columns=["userId", "movieId"]
)
print(batch)


# In[11]:


inputs = nvt_triton.convert_df_to_triton_input(["userId", "movieId"], batch, grpcclient.InferInput)

outputs = [
    grpcclient.InferRequestedOutput(col)
    for col in ["userId", "movieId", "genres__nnzs", "genres__values"]
]

MODEL_NAME_NVT = os.environ.get("MODEL_NAME_NVT", "movielens_nvt")

with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer(MODEL_NAME_NVT, inputs, request_id="1", outputs=outputs)

for col in ["userId", "movieId", "genres__nnzs", "genres__values"]:
    print(col, response.as_numpy(col), response.as_numpy(col).shape)


# You might notice that we don't need to send the genres column as an input. The reason for that is the nvt model will look up the genres for each movie as part of the `JoinExternal` op it applies. Also notice that when creating the request for the `movielens_nvt` model, we return 2 columns (values and nnzs) for the `genres` column rather than 1.

# ## END-2-END INFERENCE PIPELINE

# We will do the same, but this time we directly read in first 3 rows of the the raw `valid.parquet` file with cuDF.

# In[12]:


# read in the workflow (to get input/output schema to call triton with)
batch = df_lib.read_parquet(
    os.path.join(INPUT_DATA_DIR, "valid.parquet"), num_rows=3, columns=["userId", "movieId"]
)

print("raw data:\n", batch, "\n")

# convert the batch to a triton inputs
inputs = nvt_triton.convert_df_to_triton_input(["userId", "movieId"], batch, grpcclient.InferInput)

# placeholder variables for the output
outputs = [grpcclient.InferRequestedOutput("output")]

MODEL_NAME_ENSEMBLE = os.environ.get("MODEL_NAME_ENSEMBLE", "movielens")

# build a client to connect to our server.
# This InferenceServerClient object is what we'll be using to talk to Triton.
# make the request with tritonclient.grpc.InferInput object

with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer(MODEL_NAME_ENSEMBLE, inputs, request_id="1", outputs=outputs)

print("predicted sigmoid result:\n", response.as_numpy("output"))


# Let's send request for a larger batch size and measure the total run time and throughput.

# In[13]:


# read in the workflow (to get input/output schema to call triton with)
batch_size = 64
batch = df_lib.read_parquet(
    os.path.join(INPUT_DATA_DIR, "valid.parquet"),
    num_rows=batch_size,
    columns=["userId", "movieId"],
)

start = time()
# convert the batch to a triton inputs
inputs = nvt_triton.convert_df_to_triton_input(["userId", "movieId"], batch, grpcclient.InferInput)

# placeholder variables for the output
outputs = [grpcclient.InferRequestedOutput("output")]

MODEL_NAME_ENSEMBLE = os.environ.get("MODEL_NAME_ENSEMBLE", "movielens")

# build a client to connect to our server.
# This InferenceServerClient object is what we'll be using to talk to Triton.
# make the request with tritonclient.grpc.InferInput object

with grpcclient.InferenceServerClient("localhost:8001") as client:
    response = client.infer(MODEL_NAME_ENSEMBLE, inputs, request_id="1", outputs=outputs)

t_final = time() - start
print("predicted sigmoid result:\n", response.as_numpy("output"), "\n")

print(f"run_time(sec): {t_final} - rows: {batch_size} - inference_thru: {batch_size / t_final}")


# Let's unload all the models.

# In[14]:


triton_client.unload_model(model_name="movielens")
triton_client.unload_model(model_name="movielens_nvt")
triton_client.unload_model(model_name="movielens_tf")

