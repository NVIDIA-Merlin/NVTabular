
#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
#


# Example script for deploying a Dask-CUDA Cluster.  The
# specific options and file-paths are system dependent.
# This script was adopted from the cluster-configuration
# example at https://github.com/rapidsai/gpu-bdb
# 
# To start a single "scheduler" process and a distinct
# worker process for every visible GPU:
#
#     $ bash cluster-startup-example.sh SCHEDULER
#
# To only start the worker processes (if the scheduler
# is already up):
#
#     $ bash cluster-startup-example.sh


# Specify the protocol for cluster communication.
# Options are "tcp" and "ucx".  Note that "ucx" is
# required to leverage Infiniband (IB) and/or NVLink
PROTOCOL="ucx"

# Specify the memory limits and RMM pool size for
# each worker.  The DEVICE_MEMORY_LIMIT should be
# set to ~80% of the GPU memory capacity, and the
# POOL_SIZE should be set to ~90%+
MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT="25GB"
POOL_SIZE="30GB"

# LOCAL_DIRECTORY is used to store the scheduler file
# and scheduler/worker logs.  This directory must
# correspond to SHARED storage
LOCAL_DIRECTORY=$HOME/dask-local-directory
SCHEDULER_FILE=$LOCAL_DIRECTORY/scheduler.json
LOGDIR="$LOCAL_DIRECTORY/logs"

# WORKER_DIR corresponds to a temporary storage
# location for worker spillng.  This should be FAST
# storage, and does NOT need to be shared between nodes.
WORKER_DIR=$HOME/dask-local-directory/dask-workers/

# Purge Dask worker and log directories
if [ "$ROLE" = "SCHEDULER" ]; then
    rm -rf $LOGDIR/*
    mkdir -p $LOGDIR
    rm -rf $WORKER_DIR/*
    mkdir -p $WORKER_DIR
fi

# Purge Dask config directories
rm -rf ~/.config/dask

# Dask/distributed configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"

# Setup scheduler.
# For infiniband-enabled systems, it may be necessary to pass the
# appropriate interface (.e.g. `--interface ib0`)
ROLE=$1
if [ "$ROLE" = "SCHEDULER" ]; then
  CUDA_VISIBLE_DEVICES='0' nohup dask-scheduler --dashboard-address 8787 \
  --protocol $PROTOCOL --scheduler-file $SCHEDULER_FILE > $LOGDIR/scheduler.log 2>&1 &
fi

# Setup workers
if [ "$PROTOCOL" = "ucx" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  \
    --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --enable-tcp-over-ucx \
    --enable-nvlink --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1 &
fi

if [ "$PROTOCOL" = "tcp" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $WORKER_DIR  \
    --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY \
    --scheduler-file $SCHEDULER_FILE >> $LOGDIR/worker.log 2>&1 &
fi
