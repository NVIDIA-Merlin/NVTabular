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


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Multi-GPU with MovieLens: ETL and Training 
# 
# ## Overview
# 
# NVIDIA Merlin is a open source framework to accelerate and scale end-to-end recommender system pipelines on GPU. In this notebook, we use NVTabular, Merlin’s ETL component, to scale feature engineering and pre-processing to multiple GPUs and then perform data-parallel distributed training of a neural network on multiple GPUs with PyTorch, [Horovod](https://horovod.readthedocs.io/en/stable/), and [NCCL](https://developer.nvidia.com/nccl).
# 
# The pre-requisites for this notebook are to be familiar with NVTabular and its API:
# - You can read more about NVTabular, its API and specialized dataloaders in [Getting Started with Movielens notebooks](../getting-started-movielens).
# - You can read more about scaling NVTabular ETL in [Scaling Criteo notebooks](../scaling-criteo).
# 
# **In this notebook, we will focus only on the new information related to multi-GPU training, so please check out the other notebooks first (if you haven’t already.)**
# 
# ### Learning objectives
# 
# In this notebook, we learn how to scale ETL and deep learning taining to multiple GPUs
# - Learn to use larger than GPU/host memory datasets for ETL and training
# - Use multi-GPU or multi node for ETL with NVTabular
# - Use NVTabular dataloader to accelerate PyTorch pipelines
# - Scale PyTorch training with Horovod
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
import numpy as np

import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU

from merlin.core.utils import download_file

INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "~/nvt-examples/multigpu-movielens/data/")
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
import cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import rmm

# NVTabular
import nvtabular as nvt
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


# ## Training with PyTorch on multiGPUs
# 
# In this section, we will train a PyTorch model with multi-GPU support. In the NVTabular v0.5 release, we added multi-GPU support for NVTabular dataloaders. We will modify the [getting-started-movielens/03-Training-with-PyTorch.ipynb](../getting-started-movielens/03-Training-with-PyTorch.ipynb) to use multiple GPUs. Please review that notebook, if you have questions about the general functionality of the NVTabular dataloaders or the neural network architecture.
# 
# #### NVTabular dataloader for PyTorch
# 
# We’ve identified that the dataloader is one bottleneck in deep learning recommender systems when training pipelines with PyTorch. The normal PyTorch dataloaders cannot prepare the next training batches fast enough and therefore, the GPU is not fully utilized. 
# 
# We developed a highly customized tabular dataloader for accelerating existing pipelines in PyTorch. In our experiments, we see a speed-up by 9x of the same training workflow with NVTabular dataloader. NVTabular dataloader’s features are:
# - removing bottleneck of item-by-item dataloading
# - enabling larger than memory dataset by streaming from disk
# - reading data directly into GPU memory and remove CPU-GPU communication
# - preparing batch asynchronously in GPU to avoid CPU-GPU communication
# - supporting commonly used .parquet format
# - easy integration into existing PyTorch pipelines by using similar API
# - **supporting multi-GPU training with Horovod**
# 
# You can find more information on the dataloaders in our [blogpost](https://medium.com/nvidia-merlin/training-deep-learning-based-recommender-systems-9x-faster-with-PyTorch-cc5a2572ea49).

# #### Using Horovod with PyTorch and NVTabular
# 
# The training script below is based on [getting-started-movielens/03-Training-with-PyTorch.ipynb](../getting-started-movielens/03-Training-with-PyTorch.ipynb), with a few important changes:
# 
# - We provide several additional parameters to the `TorchAsyncItr` class, including the total number of workers `hvd.size()`, the current worker's id number `hvd.rank()`, and a function for generating random seeds `seed_fn()`. 
# 
# ```python
#     train_dataset = TorchAsyncItr(
#         ...
#         global_size=hvd.size(),
#         global_rank=hvd.rank(),
#         seed_fn=seed_fn,
#     )
# ```
# - The seed function uses Horovod to collectively generate a random seed that's shared by all workers so that they can each shuffle the dataset in a consistent way and select partitions to work on without overlap. The seed function is called by the dataloader during the shuffling process at the beginning of each epoch:
# 
# ```python
#     def seed_fn():
#         max_rand = torch.iinfo(torch.int).max // hvd.size()
# 
#         # Generate a seed fragment
#         seed_fragment = cupy.random.randint(0, max_rand)
# 
#         # Aggregate seed fragments from all Horovod workers
#         seed_tensor = torch.tensor(seed_fragment)
#         reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum)
# 
#         return reduced_seed % max_rand
# ```
# 
# - We wrap the PyTorch optimizer with Horovod's `DistributedOptimizer` class and scale the learning rate by the number of workers:
# 
# ```python
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01 * lr_scaler)
#     optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
# ```
# 
# - We broadcast the model and optimizer parameters to all workers with Horovod:
# 
# ```python
#     hvd.broadcast_parameters(model.state_dict(), root_rank=0)
#     hvd.broadcast_optimizer_state(optimizer, root_rank=0)
# ```
# 
# The rest of the script is the same as the MovieLens example in [getting-started-movielens/03-Training-with-PyTorch.ipynb](../getting-started-movielens/03-Training-with-PyTorch.ipynb). In order to run it with Horovod, we first need to write it to a file.

# In[10]:


get_ipython().run_cell_magic('writefile', "'./torch_trainer.py'", '\nimport argparse\nimport glob\nimport os\nfrom time import time\n\nimport cupy\nimport torch\n\nimport nvtabular as nvt\nfrom nvtabular.framework_utils.torch.models import Model\nfrom nvtabular.framework_utils.torch.utils import process_epoch\nfrom nvtabular.loader.torch import DLDataLoader, TorchAsyncItr\n\n# Horovod must be the last import to avoid conflicts\nimport horovod.torch as hvd  # noqa: E402, isort:skip\n\n\nparser = argparse.ArgumentParser(description="Train a multi-gpu model with Torch and Horovod")\nparser.add_argument("--dir_in", default=None, help="Input directory")\nparser.add_argument("--batch_size", default=None, help="Batch size")\nparser.add_argument("--cats", default=None, help="Categorical columns")\nparser.add_argument("--cats_mh", default=None, help="Categorical multihot columns")\nparser.add_argument("--conts", default=None, help="Continuous columns")\nparser.add_argument("--labels", default=None, help="Label columns")\nparser.add_argument("--epochs", default=1, help="Training epochs")\nargs = parser.parse_args()\n\nhvd.init()\n\ngpu_to_use = hvd.local_rank()\n\nif torch.cuda.is_available():\n    torch.cuda.set_device(gpu_to_use)\n\n\nBASE_DIR = os.path.expanduser(args.dir_in or "./data/")\nBATCH_SIZE = int(args.batch_size or 16384)  # Batch Size\nCATEGORICAL_COLUMNS = args.cats or ["movieId", "userId"]  # Single-hot\nCATEGORICAL_MH_COLUMNS = args.cats_mh or ["genres"]  # Multi-hot\nNUMERIC_COLUMNS = args.conts or []\n\n# Output from ETL-with-NVTabular\nTRAIN_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, "train", "*.parquet")))\n\nproc = nvt.Workflow.load(os.path.join(BASE_DIR, "workflow/"))\n\nEMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)\n\n\n# TensorItrDataset returns a single batch of x_cat, x_cont, y.\ndef collate_fn(x):\n    return x\n\n\n# Seed with system randomness (or a static seed)\ncupy.random.seed(None)\n\n\ndef seed_fn():\n    """\n    Generate consistent dataloader shuffle seeds across workers\n\n    Reseeds each worker\'s dataloader each epoch to get fresh a shuffle\n    that\'s consistent across workers.\n    """\n\n    max_rand = torch.iinfo(torch.int).max // hvd.size()\n\n    # Generate a seed fragment\n    seed_fragment = cupy.random.randint(0, max_rand)\n\n    # Aggregate seed fragments from all Horovod workers\n    seed_tensor = torch.tensor(seed_fragment)\n    reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum)\n\n    return reduced_seed % max_rand\n\n\ntrain_dataset = TorchAsyncItr(\n    nvt.Dataset(TRAIN_PATHS),\n    batch_size=BATCH_SIZE,\n    cats=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,\n    conts=NUMERIC_COLUMNS,\n    labels=["rating"],\n    device=gpu_to_use,\n    global_size=hvd.size(),\n    global_rank=hvd.rank(),\n    shuffle=True,\n    seed_fn=seed_fn,\n)\ntrain_loader = DLDataLoader(\n    train_dataset, batch_size=None, collate_fn=collate_fn, pin_memory=False, num_workers=0\n)\n\n\nif isinstance(EMBEDDING_TABLE_SHAPES, tuple):\n    EMBEDDING_TABLE_SHAPES_TUPLE = (\n        {\n            CATEGORICAL_COLUMNS[0]: EMBEDDING_TABLE_SHAPES[0][CATEGORICAL_COLUMNS[0]],\n            CATEGORICAL_COLUMNS[1]: EMBEDDING_TABLE_SHAPES[0][CATEGORICAL_COLUMNS[1]],\n        },\n        {CATEGORICAL_MH_COLUMNS[0]: EMBEDDING_TABLE_SHAPES[1][CATEGORICAL_MH_COLUMNS[0]]},\n    )\nelse:\n    EMBEDDING_TABLE_SHAPES_TUPLE = (\n        {\n            CATEGORICAL_COLUMNS[0]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_COLUMNS[0]],\n            CATEGORICAL_COLUMNS[1]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_COLUMNS[1]],\n        },\n        {CATEGORICAL_MH_COLUMNS[0]: EMBEDDING_TABLE_SHAPES[CATEGORICAL_MH_COLUMNS[0]]},\n    )\n\nmodel = Model(\n    embedding_table_shapes=EMBEDDING_TABLE_SHAPES_TUPLE,\n    num_continuous=0,\n    emb_dropout=0.0,\n    layer_hidden_dims=[128, 128, 128],\n    layer_dropout_rates=[0.0, 0.0, 0.0],\n).cuda()\n\nlr_scaler = hvd.size()\n\noptimizer = torch.optim.Adam(model.parameters(), lr=0.01 * lr_scaler)\n\nhvd.broadcast_parameters(model.state_dict(), root_rank=0)\nhvd.broadcast_optimizer_state(optimizer, root_rank=0)\n\noptimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())\n\nfor epoch in range(args.epochs):\n    start = time()\n    print(f"Training epoch {epoch}")\n    train_loss, y_pred, y = process_epoch(train_loader, model, train=True, optimizer=optimizer)\n    hvd.join(gpu_to_use)\n    hvd.broadcast_parameters(model.state_dict(), root_rank=0)\n    print(f"Epoch {epoch:02d}. Train loss: {train_loss:.4f}.")\n    hvd.join(gpu_to_use)\n    t_final = time() - start\n    total_rows = train_dataset.num_rows_processed\n    print(\n        f"run_time: {t_final} - rows: {total_rows} - "\n        f"epochs: {epoch} - dl_thru: {total_rows / t_final}"\n    )\n\n\nhvd.join(gpu_to_use)\nif hvd.local_rank() == 0:\n    print("Training complete")\n')


# In[11]:


get_ipython().system('horovodrun -np 2 python torch_trainer.py --dir_in $BASE_DIR --batch_size 16384')


# In[ ]:




