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


# # Scaling Criteo: Training with HugeCTR
# 
# ## Overview
# 
# HugeCTR is an open-source framework to accelerate the training of CTR estimation models on NVIDIA GPUs. It is written in CUDA C++ and highly exploits GPU-accelerated libraries such as cuBLAS, cuDNN, and NCCL.<br><br>
# HugeCTR offers multiple advantages to train deep learning recommender systems:
# 
# 1. **Speed**: HugeCTR is a highly efficient framework written C++. We experienced up to 10x speed up. HugeCTR on a NVIDIA DGX A100 system proved to be the fastest commercially available solution for training the architecture Deep Learning Recommender Model (DLRM) developed by Facebook.
# 2. **Scale**: HugeCTR supports model parallel scaling. It distributes the large embedding tables over multiple GPUs or multiple nodes. 
# 3. **Easy-to-use**: Easy-to-use Python API similar to Keras. Examples for popular deep learning recommender systems architectures (Wide&Deep, DLRM, DCN, DeepFM) are available.
# 
# HugeCTR is able to train recommender system models with larger-than-memory embedding tables by leveraging a parameter server. 
# 
# You can find more information about HugeCTR [here](https://github.com/NVIDIA/HugeCTR).
# 
# ### Learning objectives
# 
# In this notebook, we learn how to to use HugeCTR for training recommender system models
# 
# - Use **HugeCTR** to define a recommender system model
# - Train Facebook's [Deep Learning Recommendation Model](https://arxiv.org/pdf/1906.00091.pdf) with HugeCTR

# ## Training with HugeCTR

# As HugeCTR optimizes the training in CUDA++, we need to define the training pipeline and model architecture and execute it via the commandline. We will use the Python API, which is similar to Keras models.

# If you are not familiar with HugeCTR's Python API and parameters, you can read more in its GitHub repository:
# - [HugeCTR User Guide](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md)
# - [HugeCTR Python API](https://github.com/NVIDIA/HugeCTR/blob/master/docs/python_interface.md)
# - [HugeCTR example architectures](https://github.com/NVIDIA/HugeCTR/tree/master/samples)

# We will write the code to a `./model.py` file and execute it. It will create snapshot, which we will use for inference in the next notebook.

# In[2]:


get_ipython().system('ls /raid/data/criteo/test_dask/output/')


# In[3]:


import os

os.system("rm -rf ./criteo_hugectr/")
os.system("mkdir -p ./criteo_hugectr/1")


# We use `graph_to_json` to convert the model to a JSON configuration, required for the inference.

# In[4]:


get_ipython().run_cell_magic('writefile', "'./model.py'", '\nimport hugectr\nfrom mpi4py import MPI  # noqa\n\n# HugeCTR\nsolver = hugectr.CreateSolver(\n    vvgpu=[[0]],\n    max_eval_batches=100,\n    batchsize_eval=2720,\n    batchsize=2720,\n    i64_input_key=True,\n    use_mixed_precision=False,\n    repeat_dataset=True,\n)\noptimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.SGD)\nreader = hugectr.DataReaderParams(\n    data_reader_type=hugectr.DataReaderType_t.Parquet,\n    source=["/raid/data/criteo/test_dask/output/train/_file_list.txt"],\n    eval_source="/raid/data/criteo/test_dask/output/train/_file_list.txt",\n    check_type=hugectr.Check_t.Non,\n    slot_size_array=[\n        10000000,\n        10000000,\n        3014529,\n        400781,\n        11,\n        2209,\n        11869,\n        148,\n        4,\n        977,\n        15,\n        38713,\n        10000000,\n        10000000,\n        10000000,\n        584616,\n        12883,\n        109,\n        37,\n        17177,\n        7425,\n        20266,\n        4,\n        7085,\n        1535,\n        64,\n    ],\n)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(\n    hugectr.Input(\n        label_dim=1,\n        label_name="label",\n        dense_dim=13,\n        dense_name="dense",\n        data_reader_sparse_param_array=[hugectr.DataReaderSparseParam("data1", 1, False, 26)],\n    )\n)\nmodel.add(\n    hugectr.SparseEmbedding(\n        embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,\n        workspace_size_per_gpu_in_mb=6000,\n        embedding_vec_size=128,\n        combiner="sum",\n        sparse_embedding_name="sparse_embedding1",\n        bottom_name="data1",\n        optimizer=optimizer,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["dense"],\n        top_names=["fc1"],\n        num_output=512,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"])\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu1"],\n        top_names=["fc2"],\n        num_output=256,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"])\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu2"],\n        top_names=["fc3"],\n        num_output=128,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc3"], top_names=["relu3"])\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.Interaction,\n        bottom_names=["relu3", "sparse_embedding1"],\n        top_names=["interaction1"],\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["interaction1"],\n        top_names=["fc4"],\n        num_output=1024,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc4"], top_names=["relu4"])\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu4"],\n        top_names=["fc5"],\n        num_output=1024,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc5"], top_names=["relu5"])\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu5"],\n        top_names=["fc6"],\n        num_output=512,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc6"], top_names=["relu6"])\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu6"],\n        top_names=["fc7"],\n        num_output=256,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc7"], top_names=["relu7"])\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu7"],\n        top_names=["fc8"],\n        num_output=1,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,\n        bottom_names=["fc8", "label"],\n        top_names=["loss"],\n    )\n)\n\nMAX_ITER = 10000\nEVAL_INTERVAL = 3200\nmodel.compile()\nmodel.summary()\nmodel.fit(max_iter=MAX_ITER, eval_interval=EVAL_INTERVAL, display=1000, snapshot=3200)\nmodel.graph_to_json(graph_config_file="./criteo_hugectr/1/criteo.json")\n')


# In[5]:


import time

start = time.time()
get_ipython().system('python model.py')
end = time.time() - start
print(f"run_time: {end}")


# We trained the model and created snapshots.
