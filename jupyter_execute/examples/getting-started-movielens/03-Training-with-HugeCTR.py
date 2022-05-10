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
# # Getting Started MovieLens: Training with HugeCTR
# 
# In this notebook, we want to provide an overview what HugeCTR framework is, its features and benefits. We will use HugeCTR to train a basic neural network architecture.
# 
# <b>Learning Objectives</b>:
# * Adopt NVTabular workflow to provide input files to HugeCTR
# * Define HugeCTR neural network architecture
# * Train a deep learning model with HugeCTR

# ### Why using HugeCTR?
# 
# HugeCTR is a GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs).<br>
# 
# HugeCTR offers multiple advantages to train deep learning recommender systems:
# 1. **Speed**: HugeCTR is a highly efficient framework written C++. We experienced up to 10x speed up. HugeCTR on a NVIDIA DGX A100 system proved to be the fastest commercially available solution for training the architecture Deep Learning Recommender Model (DLRM) developed by Facebook.
# 2. **Scale**: HugeCTR supports model parallel scaling. It distributes the large embedding tables over multiple GPUs or multiple nodes. 
# 3. **Easy-to-use**: Easy-to-use Python API similar to Keras. Examples for popular deep learning recommender systems architectures (Wide&Deep, DLRM, DCN, DeepFM) are available.

# ### Other Features of HugeCTR
# 
# HugeCTR is designed to scale deep learning models for recommender systems. It provides a list of other important features:
# * Proficiency in oversubscribing models to train embedding tables with single nodes that don’t fit within the GPU or CPU memory (only required embeddings are prefetched from a parameter server per batch)
# * Asynchronous and multithreaded data pipelines
# * A highly optimized data loader.
# * Supported data formats such as parquet and binary
# * Integration with Triton Inference Server for deployment to production

# ### Getting Started

# In this example, we will train a neural network with HugeCTR. We will use preprocessed datasets generated via NVTabular in `02-ETL-with-NVTabular` notebook.

# In[2]:


# External dependencies
import os
import nvtabular as nvt


# We define our base directory, containing the data.

# In[3]:


# path to preprocessed data
INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/movielens/data/")
)

# path to save the models
MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", os.path.expanduser("~/nvt-examples/"))


# Let's load our saved workflow from the `02-ETL-with-NVTabular` notebook.

# In[4]:


workflow = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow"))


# In[5]:


workflow.output_dtypes


# Note: We do not have numerical output columns

# Let's clear existing directory and create the output folders.

# In[6]:


MODEL_DIR = os.path.join(INPUT_DATA_DIR, "model/movielens_hugectr/")
get_ipython().system('rm -rf {MODEL_DIR}')
get_ipython().system('mkdir -p {MODEL_DIR}"1"')


# ## Scaling Accelerated training with HugeCTR

# HugeCTR is a deep learning framework dedicated to recommendation systems. It is written in CUDA C++. As HugeCTR optimizes the training in CUDA++, we need to define the training pipeline and model architecture and execute it via the commandline. We will use the Python API, which is similar to Keras models.

# HugeCTR has three main components:
# * Solver: Specifies various details such as active GPU list, batchsize, and model_file
# * Optimizer: Specifies the type of optimizer and its hyperparameters
# * DataReader: Specifies the training/evaluation data
# * Model: Specifies embeddings, and dense layers. Note that embeddings must precede the dense layers

# **Solver**
# 
# Let's take a look on the parameter for the `Solver`. We should be familiar from other frameworks for the hyperparameter.
# 
# ```
# solver = hugectr.CreateSolver(
# - vvgpu: GPU indices used in the training process, which has two levels. For example: [[0,1],[1,2]] indicates that two physical nodes (each physical node can have multiple NUMA nodes) are used. In the first node, GPUs 0 and 1 are used while GPUs 1 and 2 are used for the second node. It is also possible to specify non-continuous GPU indices such as [0, 2, 4, 7].
# - batchsize: Minibatch size used in training
# - max_eval_batches: Maximum number of batches used in evaluation. It is recommended that the number is equal to or bigger than the actual number of bathces in the evaluation dataset.
# On the other hand, with num_epochs, HugeCTR stops the evaluation if all the evaluation data is consumed    
# - batchsize_eval: Minibatch size used in evaluation. The default value is 2048. Note that batchsize here is the global batch size across gpus and nodes, not per worker batch size.
# - mixed_precision: Enables mixed precision training with the scaler specified here. Only 128,256, 512, and 1024 scalers are supported
# )
# ```

# **Optimizer**
# 
# The optimizer is the algorithm to update the model parameters. HugeCTR supports the common algorithms.
# 
# 
# ```
# optimizer = CreateOptimizer(
# - optimizer_type: Optimizer algorithm - Adam, MomentumSGD, Nesterov, and SGD 
# - learning_rate: Learning Rate for optimizer
# )
# ```

# **DataReader**
# 
# The data reader defines the training and evaluation dataset.
# 
# 
# ```
# reader = hugectr.DataReaderParams(
# - data_reader_type: Data format to read
# - source: The training dataset file list. IMPORTANT: This should be a list
# - eval_source: The evaluation dataset file list.
# - check_type: The data error detection mechanism (Sum: Checksum, None: no detection).
# - slot_size_array: The list of categorical feature cardinalities
# )
# ```

# **Model**
# 
# We initialize the model with the solver, optimizer and data reader:
# 
# ```
# model = hugectr.Model(solver, reader, optimizer)
# ```
# 
# We can add multiple layers to the model with `model.add` function. We will focus on:
# - `Input` defines the input data
# - `SparseEmbedding` defines the embedding layer
# - `DenseLayer` defines dense layers, such as fully connected, ReLU, BatchNorm, etc.
# 
# **HugeCTR organizes the layers by names. For each layer, we define the input and output names.**

# Input layer:
# 
# This layer is required to define the input data.
# 
# ```
# hugectr.Input(
#     label_dim: Number of label columns
#     label_name: Name of label columns in network architecture
#     dense_dim: Number of continuous columns
#     dense_name: Name of contiunous columns in network architecture
#     data_reader_sparse_param_array: Configuration how to read sparse data and its names
# )
# ```
# 
# SparseEmbedding:
# 
# This layer defines embedding table
# 
# ```
# hugectr.SparseEmbedding(
#     embedding_type: Different embedding options to distribute embedding tables 
#     workspace_size_per_gpu_in_mb: Maximum embedding table size in MB
#     embedding_vec_size: Embedding vector size
#     combiner: Intra-slot reduction op
#     sparse_embedding_name: Layer name
#     bottom_name: Input layer names
#     optimizer: Optimizer to use
# )
# ```
# 
# DenseLayer:
# 
# This layer is copied to each GPU and is normally used for the MLP tower.
# 
# ```
# hugectr.DenseLayer(
#     layer_type: Layer type, such as FullyConnected, Reshape, Concat, Loss, BatchNorm, etc.
#     bottom_names: Input layer names
#     top_names: Layer name
#     ...: Depending on the layer type additional parameter can be defined
# )
# ```
# 
# This is only a short introduction in the API. You can read more in the official docs: [Python Interface](https://github.com/NVIDIA/HugeCTR/blob/master/docs/python_interface.md) and [Layer Book](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_layer_book.md)

# ## Let's define our model
# 
# We walked through the documentation, but it is useful to understand the API. Finally, we can define our model. We will write the model to `./model.py` and execute it afterwards.

# We need the cardinalities of each categorical feature to assign as `slot_size_array` in the model below.

# In[7]:


from nvtabular.ops import get_embedding_sizes

embeddings = get_embedding_sizes(workflow)
print(embeddings)


# We use `graph_to_json` to convert the model to a JSON configuration, required for the inference.

# In[8]:


import hugectr
from mpi4py import MPI  # noqa

solver = hugectr.CreateSolver(
    vvgpu=[[0]],
    batchsize=2048,
    batchsize_eval=2048,
    max_eval_batches=160,
    i64_input_key=True,
    use_mixed_precision=False,
    repeat_dataset=True,
)
optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam)
reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.Parquet,
    source=[INPUT_DATA_DIR + "train/_file_list.txt"],
    eval_source=INPUT_DATA_DIR + "valid/_file_list.txt",
    check_type=hugectr.Check_t.Non,
    slot_size_array=[162542, 56586, 21],
)


model = hugectr.Model(solver, reader, optimizer)

model.add(
    hugectr.Input(
        label_dim=1,
        label_name="label",
        dense_dim=0,
        dense_name="dense",
        data_reader_sparse_param_array=[
            hugectr.DataReaderSparseParam("data1", nnz_per_slot=10, is_fixed_length=False, slot_num=3)
        ],
    )
)
model.add(
    hugectr.SparseEmbedding(
        embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,
        workspace_size_per_gpu_in_mb=200,
        embedding_vec_size=16,
        combiner="sum",
        sparse_embedding_name="sparse_embedding1",
        bottom_name="data1",
        optimizer=optimizer,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=["sparse_embedding1"],
        top_names=["reshape1"],
        leading_dim=48,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["reshape1"],
        top_names=["fc1"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU,
        bottom_names=["fc1"],
        top_names=["relu1"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu1"],
        top_names=["fc2"],
        num_output=128,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.ReLU,
        bottom_names=["fc2"],
        top_names=["relu2"],
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=["relu2"],
        top_names=["fc3"],
        num_output=1,
    )
)
model.add(
    hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
        bottom_names=["fc3", "label"],
        top_names=["loss"],
    )
)


# In[9]:


model.compile()
model.summary()
model.fit(max_iter=2000, display=100, eval_interval=200, snapshot=1900)
model.graph_to_json(graph_config_file=MODEL_DIR + "1/movielens.json")


# After training terminates, we can see that multiple `.model` files and folders are generated. We need to move them inside `1` folder under the `movielens_hugectr` folder. 

# In[10]:


get_ipython().system('mv *.model {MODEL_DIR}')

