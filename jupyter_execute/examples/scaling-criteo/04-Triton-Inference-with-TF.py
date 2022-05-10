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


# # Scaling Criteo: Triton Inference with TensorFlow
# 
# ## Overview
# 
# The last step is to deploy the ETL workflow and saved model to production. In the production setting, we want to transform the input data as during training (ETL). We need to apply the same mean/std for continuous features and use the same categorical mapping to convert the categories to continuous integer before we use the deep learning model for a prediction. Therefore, we deploy the NVTabular workflow with the TensorFlow model as an ensemble model to Triton Inference. The ensemble model garantuees that the same transformation are applied to the raw inputs.
# 
# <img src='./imgs/triton-tf.png' width="25%">
# 
# ### Learning objectives
# 
# In this notebook, we learn how to deploy our models to production
# 
# - Use **NVTabular** to generate config and model files for Triton Inference Server
# - Deploy an ensemble of NVTabular workflow and TensorFlow model
# - Send example request to Triton Inference Server

# ## Inference with Triton and TensorFlow
# 
# First, we need to generate the Triton Inference Server configurations and save the models in the correct format. In the previous notebooks [02-ETL-with-NVTabular](./02-ETL-with-NVTabular.ipynb) and [03-Training-with-TF](./03-Training-with-TF.ipynb) we saved the NVTabular workflow and TensorFlow model to disk. We will load them.

# ### Saving Ensemble Model for Triton Inference Server

# In[2]:


import os

import tensorflow as tf
import nvtabular as nvt


# In[3]:


BASE_DIR = os.environ.get("BASE_DIR", "/raid/data/criteo")
input_path = os.path.join(BASE_DIR, "test_dask/output")


# In[4]:


workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))


# In[5]:


model = tf.keras.models.load_model(os.path.join(input_path, "model.savedmodel"))


# TensorFlow expect the Integer as `int32` datatype. Therefore, we need to define the NVTabular output datatypes to `int32` for categorical features.

# In[6]:


for key in workflow.output_dtypes.keys():
    if key.startswith("C"):
        workflow.output_dtypes[key] = "int32"


# NVTabular provides an easy function to deploy the ensemble model for Triton Inference Server.

# In[7]:


from nvtabular.inference.triton import export_tensorflow_ensemble


# In[10]:


export_tensorflow_ensemble(model, workflow, "criteo", "/models", ["label"])


# We can take a look on the generated files.

# In[11]:


get_ipython().system('tree /models')


# ### Loading Ensemble Model with Triton Inference Server
# 
# We have only saved the models for Triton Inference Server. We started Triton Inference Server in explicit mode, meaning that we need to send a request that Triton will load the ensemble model.

# First, we restart this notebook to free the GPU memory.

# In[12]:


import IPython

app = IPython.Application.instance()
app.kernel.do_shutdown(True)


# We define the BASE_DIR again.

# In[2]:


import os

BASE_DIR = os.environ.get("BASE_DIR", "/raid/data/criteo")


# We connect to the Triton Inference Server.

# In[6]:


import tritonhttpclient

try:
    triton_client = tritonhttpclient.InferenceServerClient(url="triton:8000", verbose=True)
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))


# We deactivate warnings.

# In[7]:


import warnings

warnings.filterwarnings("ignore")


# We check if the server is alive.

# In[8]:


triton_client.is_server_live()


# We check the available models in the repositories:
# - criteo: Ensemble 
# - criteo_nvt: NVTabular 
# - criteo_tf: TensorFlow model

# In[9]:


triton_client.get_model_repository_index()


# We load the ensembled model.

# In[10]:


get_ipython().run_cell_magic('time', '', '\ntriton_client.load_model(model_name="criteo")\n')


# ### Example Request to Triton Inference Server
# 
# Now, the models are loaded and we can create a sample request. We read an example **raw batch** for inference.

# In[11]:


# Get dataframe library - cudf or pandas
from merlin.core.dispatch import get_lib

df_lib = get_lib()

# read in the workflow (to get input/output schema to call triton with)
batch_path = os.path.join(BASE_DIR, "converted/criteo")
batch = df_lib.read_parquet(os.path.join(batch_path, "*.parquet"), num_rows=3)
batch = batch[[x for x in batch.columns if x != "label"]]
print(batch)


# We prepare the batch for inference by using correct column names and data types. We use the same datatypes as defined in our dataframe.

# In[12]:


batch.dtypes


# In[13]:


import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np

inputs = []

col_names = list(batch.columns)
col_dtypes = [np.int32] * len(col_names)

for i, col in enumerate(batch.columns):
    d = batch[col].fillna(0).values_host.astype(col_dtypes[i])
    d = d.reshape(len(d), 1)
    inputs.append(httpclient.InferInput(col_names[i], d.shape, np_to_triton_dtype(col_dtypes[i])))
    inputs[i].set_data_from_numpy(d)


# We send the request to the triton server and collect the last output.

# In[14]:


# placeholder variables for the output
outputs = [httpclient.InferRequestedOutput("output")]

# build a client to connect to our server.
# This InferenceServerClient object is what we'll be using to talk to Triton.
# make the request with tritonclient.http.InferInput object
response = triton_client.infer("criteo", inputs, request_id="1", outputs=outputs)

print("predicted softmax result:\n", response.as_numpy("output"))


# Let's unload the model. We need to unload each model.

# In[16]:


triton_client.unload_model(model_name="criteo")
triton_client.unload_model(model_name="criteo_nvt")
triton_client.unload_model(model_name="criteo_tf")

