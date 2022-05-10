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
# # Multi-GPU Training with TensorFlow on MovieLens
# 
# ## Overview
# 
# NVIDIA Merlin is a open source framework to accelerate and scale end-to-end recommender system pipelines on GPU. In this notebook, we use NVTabular, Merlin’s ETL component, to scale feature engineering and pre-processing to multiple GPUs and then perform data-parallel distributed training of a neural network on multiple GPUs with TensorFlow, [Horovod](https://horovod.readthedocs.io/en/stable/), and [NCCL](https://developer.nvidia.com/nccl).
# 
# The pre-requisites for this notebook are to be familiar with NVTabular and its API:
# - You can read more about NVTabular, its API and specialized dataloaders in [Getting Started with Movielens notebooks](https://nvidia-merlin.github.io/NVTabular/main/examples/getting-started-movielens/index.html).
# - You can read more about scaling NVTabular ETL in [Scaling Criteo notebooks](https://nvidia-merlin.github.io/NVTabular/main/examples/scaling-criteo/index.html).
# 
# **In this notebook, we will focus only on the new information related to multi-GPU training, so please check out the other notebooks first (if you haven’t already.)**
# 
# ### Learning objectives
# 
# In this notebook, we learn how to scale ETL and deep learning taining to multiple GPUs
# - Learn to use larger than GPU/host memory datasets for ETL and training
# - Use multi-GPU or multi node for ETL with NVTabular
# - Use NVTabular dataloader to accelerate TensorFlow pipelines
# - Scale TensorFlow training with Horovod
# 
# ### Dataset
# 
# In this notebook, we use the [MovieLens25M](https://grouplens.org/datasets/movielens/25m/) dataset. It is popular for recommender systems and is used in academic publications. The dataset contains 25M movie ratings for 62,000 movies given by 162,000 users. Many projects use only the user/item/rating information of MovieLens, but the original dataset provides metadata for the movies, as well.
# 
# Note: We are using the MovieLens 25M dataset in this example for simplicity, although the dataset is not large enough to require multi-GPU training. However, the functionality demonstrated in this notebook can be easily extended to scale recommender pipelines for larger datasets in the same way.
# 
# ### Tools
# 
# - [Horovod](https://horovod.readthedocs.io/en/stable/) is a distributed deep learning framework that provides tools for multi-GPU optimization.
# - The [NVIDIA Collective Communication Library (NCCL)](https://developer.nvidia.com/nccl) provides the underlying GPU-based implementations of the [allgather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#allgather) and [allreduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#allreduce) cross-GPU communication operations.

# ## Download and Convert
# 
# First, we will download and convert the dataset to Parquet. This section is based on [01-Download-Convert.ipynb](../getting-started-movielens/01-Download-Convert.ipynb).

# #### Download

# In[2]:


# External dependencies
import os
import pathlib

import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU

from merlin.core.utils import download_file

INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", "~/nvt-examples/multigpu-movielens/data/"
)
BASE_DIR = pathlib.Path(INPUT_DATA_DIR).expanduser()
zip_path = pathlib.Path(BASE_DIR, "ml-25m.zip")
download_file(
    "http://files.grouplens.org/datasets/movielens/ml-25m.zip", zip_path, redownload=False
)


# #### Convert

# In[3]:


movies = cudf.read_csv(pathlib.Path(BASE_DIR, "ml-25m", "movies.csv"))
movies["genres"] = movies["genres"].str.split("|")
movies = movies.drop("title", axis=1)
movies.to_parquet(pathlib.Path(BASE_DIR, "ml-25m", "movies_converted.parquet"))


# #### Split into train and validation datasets

# In[4]:


ratings = cudf.read_csv(pathlib.Path(BASE_DIR, "ml-25m", "ratings.csv"))
ratings = ratings.drop("timestamp", axis=1)

# shuffle the dataset
ratings = ratings.sample(len(ratings), replace=False)
# split the train_df as training and validation data sets.
num_valid = int(len(ratings) * 0.2)
train = ratings[:-num_valid]
valid = ratings[-num_valid:]

train.to_parquet(pathlib.Path(BASE_DIR, "train.parquet"))
valid.to_parquet(pathlib.Path(BASE_DIR, "valid.parquet"))


# ## ETL with NVTabular
# 
# We finished downloading and converting the dataset. We will preprocess and engineer features with NVTabular on multiple GPUs. You can read more
# - about NVTabular's features and API in [getting-started-movielens/02-ETL-with-NVTabular.ipynb](../getting-started-movielens/02-ETL-with-NVTabular.ipynb).
# - scaling NVTabular ETL to multiple GPUs [scaling-criteo/02-ETL-with-NVTabular.ipynb](../scaling-criteo/02-ETL-with-NVTabular.ipynb).

# #### Deploy a Distributed-Dask Cluster
# 
# This section is based on [scaling-criteo/02-ETL-with-NVTabular.ipynb](../scaling-criteo/02-ETL-with-NVTabular.ipynb) and [multi-gpu-toy-example/multi-gpu_dask.ipynb](../multi-gpu-toy-example/multi-gpu_dask.ipynb)

# In[5]:


# Standard Libraries
import shutil

# External Dependencies
import cupy as cp
import numpy as np
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import rmm

# NVTabular
import nvtabular as nvt
import nvtabular.ops as ops
from merlin.io import Shuffle
from merlin.core.utils import device_mem_size


# In[6]:


# define some information about where to get our data
input_path = pathlib.Path(BASE_DIR, "converted", "movielens")
dask_workdir = pathlib.Path(BASE_DIR, "test_dask", "workdir")
output_path = pathlib.Path(BASE_DIR, "test_dask", "output")
stats_path = pathlib.Path(BASE_DIR, "test_dask", "stats")

# Make sure we have a clean worker space for Dask
if pathlib.Path.is_dir(dask_workdir):
    shutil.rmtree(dask_workdir)
dask_workdir.mkdir(parents=True)

# Make sure we have a clean stats space for Dask
if pathlib.Path.is_dir(stats_path):
    shutil.rmtree(stats_path)
stats_path.mkdir(parents=True)

# Make sure we have a clean output path
if pathlib.Path.is_dir(output_path):
    shutil.rmtree(output_path)
output_path.mkdir(parents=True)

# Get device memory capacity
capacity = device_mem_size(kind="total")


# In[7]:


# Deploy a Single-Machine Multi-GPU Cluster
protocol = "tcp"  # "tcp" or "ucx"
visible_devices = "0,1"  # Delect devices to place workers
device_spill_frac = 0.5  # Spill GPU-Worker memory to host at this limit.
# Reduce if spilling fails to prevent
# device memory errors.
cluster = None  # (Optional) Specify existing scheduler port
if cluster is None:
    cluster = LocalCUDACluster(
        protocol=protocol,
        CUDA_VISIBLE_DEVICES=visible_devices,
        local_directory=dask_workdir,
        device_memory_limit=capacity * device_spill_frac,
    )

# Create the distributed client
client = Client(cluster)
client


# In[8]:


# Initialize RMM pool on ALL workers
def _rmm_pool():
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=None,  # Use default size
    )


client.run(_rmm_pool)


# #### Defining our Preprocessing Pipeline
# 
# This subsection is based on [getting-started-movielens/02-ETL-with-NVTabular.ipynb](../getting-started-movielens/02-ETL-with-NVTabular.ipynb).

# In[9]:


movies = cudf.read_parquet(pathlib.Path(BASE_DIR, "ml-25m", "movies_converted.parquet"))
joined = ["userId", "movieId"] >> nvt.ops.JoinExternal(movies, on=["movieId"])
cat_features = joined >> nvt.ops.Categorify()
ratings = nvt.ColumnSelector(["rating"]) >> nvt.ops.LambdaOp(lambda col: (col > 3).astype("int8"), dtype=np.int8)
output = cat_features + ratings
workflow = nvt.Workflow(output)
get_ipython().system('rm -rf $BASE_DIR/train')
get_ipython().system('rm -rf $BASE_DIR/valid')
train_iter = nvt.Dataset([str(pathlib.Path(BASE_DIR, "train.parquet"))], part_size="100MB")
valid_iter = nvt.Dataset([str(pathlib.Path(BASE_DIR, "valid.parquet"))], part_size="100MB")
workflow.fit(train_iter)
workflow.save(str(pathlib.Path(BASE_DIR, "workflow")))
shuffle = Shuffle.PER_WORKER  # Shuffle algorithm
out_files_per_proc = 4  # Number of output files per worker
workflow.transform(train_iter).to_parquet(
    output_path=pathlib.Path(BASE_DIR, "train"),
    shuffle=shuffle,
    out_files_per_proc=out_files_per_proc,
)
workflow.transform(valid_iter).to_parquet(
    output_path=pathlib.Path(BASE_DIR, "valid"),
    shuffle=shuffle,
    out_files_per_proc=out_files_per_proc,
)

client.shutdown()
cluster.close()


# ## Training with TensorFlow on multiGPUs
# 
# In this section, we will train a TensorFlow model with multi-GPU support. In the NVTabular v0.5 release, we added multi-GPU support for NVTabular dataloaders. We will modify the [getting-started-movielens/03-Training-with-TF.ipynb](../getting-started-movielens/03-Training-with-TF.ipynb) to use multiple GPUs. Please review that notebook, if you have questions about the general functionality of the NVTabular dataloaders or the neural network architecture.
# 
# #### NVTabular dataloader for TensorFlow
# 
# We’ve identified that the dataloader is one bottleneck in deep learning recommender systems when training pipelines with TensorFlow. The normal TensorFlow dataloaders cannot prepare the next training batches fast enough and therefore, the GPU is not fully utilized. 
# 
# We developed a highly customized tabular dataloader for accelerating existing pipelines in TensorFlow. In our experiments, we see a speed-up by 9x of the same training workflow with NVTabular dataloader. NVTabular dataloader’s features are:
# - removing bottleneck of item-by-item dataloading
# - enabling larger than memory dataset by streaming from disk
# - reading data directly into GPU memory and remove CPU-GPU communication
# - preparing batch asynchronously in GPU to avoid CPU-GPU communication
# - supporting commonly used .parquet format
# - easy integration into existing TensorFlow pipelines by using similar API - works with tf.keras models
# - **supporting multi-GPU training with Horovod**
# 
# You can find more information on the dataloaders in our [blogpost](https://medium.com/nvidia-merlin/training-deep-learning-based-recommender-systems-9x-faster-with-tensorflow-cc5a2572ea49).

# #### Using Horovod with Tensorflow and NVTabular
# 
# The training script below is based on [getting-started-movielens/03-Training-with-TF.ipynb](../getting-started-movielens/03-Training-with-TF.ipynb), with a few important changes:
# 
# - We provide several additional parameters to the `KerasSequenceLoader` class, including the total number of workers `hvd.size()`, the current worker's id number `hvd.rank()`, and a function for generating random seeds `seed_fn()`. 
# 
# ```python
#     train_dataset_tf = KerasSequenceLoader(
#         ...
#         global_size=hvd.size(),
#         global_rank=hvd.rank(),
#         seed_fn=seed_fn,
#     )
# 
# ```
# - The seed function uses Horovod to collectively generate a random seed that's shared by all workers so that they can each shuffle the dataset in a consistent way and select partitions to work on without overlap. The seed function is called by the dataloader during the shuffling process at the beginning of each epoch:
# 
# ```python
#     def seed_fn():
#         min_int, max_int = tf.int32.limits
#         max_rand = max_int // hvd.size()
# 
#         # Generate a seed fragment on each worker
#         seed_fragment = cupy.random.randint(0, max_rand).get()
# 
#         # Aggregate seed fragments from all Horovod workers
#         seed_tensor = tf.constant(seed_fragment)
#         reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum) 
# 
#         return reduced_seed % max_rand
# ```
# 
# - We wrap the TensorFlow optimizer with Horovod's `DistributedOptimizer` class and scale the learning rate by the number of workers:
# 
# ```python
#     opt = tf.keras.optimizers.SGD(0.01 * hvd.size())
#     opt = hvd.DistributedOptimizer(opt)
# ```
# 
# - We wrap the TensorFlow gradient tape with Horovod's `DistributedGradientTape` class:
# 
# ```python
#     with tf.GradientTape() as tape:
#         ...
#     tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True)
# ```
# 
# - After the first batch, we broadcast the model and optimizer parameters to all workers with Horovod:
# 
# ```python
#     # Note: broadcast should be done after the first gradient step to
#     # ensure optimizer initialization.
#     if first_batch:
#         hvd.broadcast_variables(model.variables, root_rank=0)
#         hvd.broadcast_variables(opt.variables(), root_rank=0)
# ```
# 
# - We only save checkpoints from the first worker to avoid multiple workers trying to write to the same files:
# 
# ```python
#     if hvd.rank() == 0:
#         checkpoint.save(checkpoint_dir)
# ```
# 
# The rest of the script is the same as the MovieLens example in [getting-started-movielens/03-Training-with-TF.ipynb](../getting-started-movielens/03-Training-with-TF.ipynb). In order to run it with Horovod, we first need to write it to a file.

# In[10]:


get_ipython().run_cell_magic('writefile', "'./tf_trainer.py'", '\n# External dependencies\nimport argparse\nimport glob\nimport os\n\nimport cupy\n\n# we can control how much memory to give tensorflow with this environment variable\n# IMPORTANT: make sure you do this before you initialize TF\'s runtime, otherwise\n# TF will have claimed all free GPU memory\nos.environ["TF_MEMORY_ALLOCATION"] = "0.3"  # fraction of free memory\n\nimport nvtabular as nvt  # noqa: E402 isort:skip\nfrom nvtabular.framework_utils.tensorflow import layers  # noqa: E402 isort:skip\nfrom nvtabular.loader.tensorflow import KerasSequenceLoader  # noqa: E402 isort:skip\n\nimport tensorflow as tf  # noqa: E402 isort:skip\nimport horovod.tensorflow as hvd  # noqa: E402 isort:skip\n\nparser = argparse.ArgumentParser(description="Process some integers.")\nparser.add_argument("--dir_in", default=None, help="Input directory")\nparser.add_argument("--batch_size", default=None, help="batch size")\nparser.add_argument("--cats", default=None, help="categorical columns")\nparser.add_argument("--cats_mh", default=None, help="categorical multihot columns")\nparser.add_argument("--conts", default=None, help="continuous columns")\nparser.add_argument("--labels", default=None, help="continuous columns")\nargs = parser.parse_args()\n\n\nBASE_DIR = args.dir_in or "./data/"\nBATCH_SIZE = int(args.batch_size or 16384)  # Batch Size\nCATEGORICAL_COLUMNS = args.cats or ["movieId", "userId"]  # Single-hot\nCATEGORICAL_MH_COLUMNS = args.cats_mh or ["genres"]  # Multi-hot\nNUMERIC_COLUMNS = args.conts or []\nTRAIN_PATHS = sorted(\n    glob.glob(os.path.join(BASE_DIR, "train/*.parquet"))\n)  # Output from ETL-with-NVTabular\nhvd.init()\n\n# Seed with system randomness (or a static seed)\ncupy.random.seed(None)\n\n\ndef seed_fn():\n    """\n    Generate consistent dataloader shuffle seeds across workers\n\n    Reseeds each worker\'s dataloader each epoch to get fresh a shuffle\n    that\'s consistent across workers.\n    """\n    min_int, max_int = tf.int32.limits\n    max_rand = max_int // hvd.size()\n\n    # Generate a seed fragment on each worker\n    seed_fragment = cupy.random.randint(0, max_rand).get()\n\n    # Aggregate seed fragments from all Horovod workers\n    seed_tensor = tf.constant(seed_fragment)\n    reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum)\n\n    return reduced_seed % max_rand\n\n\nproc = nvt.Workflow.load(os.path.join(BASE_DIR, "workflow/"))\nEMBEDDING_TABLE_SHAPES, MH_EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)\nEMBEDDING_TABLE_SHAPES.update(MH_EMBEDDING_TABLE_SHAPES)\n\ntrain_dataset_tf = KerasSequenceLoader(\n    TRAIN_PATHS,  # you could also use a glob pattern\n    batch_size=BATCH_SIZE,\n    label_names=["rating"],\n    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,\n    cont_names=NUMERIC_COLUMNS,\n    engine="parquet",\n    shuffle=True,\n    buffer_size=0.06,  # how many batches to load at once\n    parts_per_chunk=1,\n    global_size=hvd.size(),\n    global_rank=hvd.rank(),\n    seed_fn=seed_fn,\n)\ninputs = {}  # tf.keras.Input placeholders for each feature to be used\nemb_layers = []  # output of all embedding layers, which will be concatenated\nfor col in CATEGORICAL_COLUMNS:\n    inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))\n# Note that we need two input tensors for multi-hot categorical features\nfor col in CATEGORICAL_MH_COLUMNS:\n    inputs[col] = \\\n        (tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,)),\n         tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,)))\nfor col in CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS:\n    emb_layers.append(\n        tf.feature_column.embedding_column(\n            tf.feature_column.categorical_column_with_identity(\n                col, EMBEDDING_TABLE_SHAPES[col][0]\n            ),  # Input dimension (vocab size)\n            EMBEDDING_TABLE_SHAPES[col][1],  # Embedding output dimension\n        )\n    )\nemb_layer = layers.DenseFeatures(emb_layers)\nx_emb_output = emb_layer(inputs)\nx = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)\nx = tf.keras.layers.Dense(128, activation="relu")(x)\nx = tf.keras.layers.Dense(128, activation="relu")(x)\nx = tf.keras.layers.Dense(1, activation="sigmoid")(x)\nmodel = tf.keras.Model(inputs=inputs, outputs=x)\nloss = tf.losses.BinaryCrossentropy()\nopt = tf.keras.optimizers.SGD(0.01 * hvd.size())\nopt = hvd.DistributedOptimizer(opt)\ncheckpoint_dir = "./checkpoints"\ncheckpoint = tf.train.Checkpoint(model=model, optimizer=opt)\n\n\n@tf.function(experimental_relax_shapes=True)\ndef training_step(examples, labels, first_batch):\n    with tf.GradientTape() as tape:\n        probs = model(examples, training=True)\n        loss_value = loss(labels, probs)\n    # Horovod: add Horovod Distributed GradientTape.\n    tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True)\n    grads = tape.gradient(loss_value, model.trainable_variables)\n    opt.apply_gradients(zip(grads, model.trainable_variables))\n    # Horovod: broadcast initial variable states from rank 0 to all other processes.\n    # This is necessary to ensure consistent initialization of all workers when\n    # training is started with random weights or restored from a checkpoint.\n    #\n    # Note: broadcast should be done after the first gradient step to ensure optimizer\n    # initialization.\n    if first_batch:\n        hvd.broadcast_variables(model.variables, root_rank=0)\n        hvd.broadcast_variables(opt.variables(), root_rank=0)\n    return loss_value\n\n\n# Horovod: adjust number of steps based on number of GPUs.\nfor batch, (examples, labels) in enumerate(train_dataset_tf):\n    loss_value = training_step(examples, labels, batch == 0)\n    if batch % 100 == 0 and hvd.local_rank() == 0:\n        print("Step #%d\\tLoss: %.6f" % (batch, loss_value))\nhvd.join()\n\n# Horovod: save checkpoints only on worker 0 to prevent other workers from\n# corrupting it.\nif hvd.rank() == 0:\n    checkpoint.save(checkpoint_dir)\n')


# We'll also need a small wrapper script to check environment variables set by the Horovod runner to see which rank we'll be assigned, in order to set CUDA_VISIBLE_DEVICES properly for each worker:

# In[11]:


get_ipython().run_cell_magic('writefile', "'./hvd_wrapper.sh'", '\n#!/bin/bash\n\n# Get local process ID from OpenMPI or alternatively from SLURM\nif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then\n    if [ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]; then\n        LOCAL_RANK="${OMPI_COMM_WORLD_LOCAL_RANK}"\n    elif [ -n "${SLURM_LOCALID:-}" ]; then\n        LOCAL_RANK="${SLURM_LOCALID}"\n    fi\n    export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}\nfi\n\nexec "$@"\n')


# OpenMPI and Slurm are tools for running distributed computed jobs. In this example, we’re using OpenMPI, but depending on the environment you run distributed training jobs in, you may need to check slightly different environment variables to find the total number of workers (global size) and each process’s worker number (global rank.)
# 
# Why do we have to check environment variables instead of using `hvd.rank()` and `hvd.local_rank()`? NVTabular does some GPU configuration when imported and needs to be imported before Horovod to avoid conflicts. We need to set GPU visibility before NVTabular is imported (when Horovod isn’t yet available) so that multiple processes don’t each try to configure all the GPUs, so as a workaround, we “cheat” and peek at environment variables set by horovodrun to decide which GPU each process should use.

# In[12]:


get_ipython().system('horovodrun -np 2 sh hvd_wrapper.sh python tf_trainer.py --dir_in $BASE_DIR --batch_size 16384')

