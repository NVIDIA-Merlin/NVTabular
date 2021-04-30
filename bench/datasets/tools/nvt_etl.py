#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

import contextlib
import os
import shutil
import warnings

import numpy as np
import rmm
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from nvtabular import Dataset, Workflow
from nvtabular import io as nvt_io
from nvtabular import ops
from nvtabular.utils import _pynvml_mem_size, device_mem_size, get_rmm_size


def setup_rmm_pool(client, device_pool_size):
    # Initialize an RMM pool allocator.
    # Note: RMM may require the pool size to be a multiple of 256.
    device_pool_size = get_rmm_size(device_pool_size)
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=device_pool_size)
    return None


@contextlib.contextmanager
def managed_client(dask_workdir, devices, device_limit, protocol):
    if protocol == "tcp":
        client = Client(
            LocalCUDACluster(
                protocol=protocol,
                n_workers=len(devices.split(",")),
                CUDA_VISIBLE_DEVICES=devices,
                device_memory_limit=device_limit,
                local_directory=dask_workdir,
            )
        )
    else:
        client = Client(
            LocalCUDACluster(
                protocol=protocol,
                n_workers=len(devices.split(",")),
                CUDA_VISIBLE_DEVICES=devices,
                enable_nvlink=True,
                device_memory_limit=device_limit,
                local_directory=dask_workdir,
            )
        )
    try:
        yield client
    finally:
        client.shutdown()


def nvt_etl(
    data_path,
    out_path,
    devices,
    protocol,
    device_limit_frac,
    device_pool_frac,
    part_mem_frac,
    cats,
    conts,
    labels,
    out_files_per_proc,
):
    # Set up data paths
    input_path = data_path[:-1] if data_path[-1] == "/" else data_path
    base_dir = out_path[:-1] if out_path[-1] == "/" else out_path
    dask_workdir = os.path.join(base_dir, "workdir")
    output_path = os.path.join(base_dir, "output")
    stats_path = os.path.join(base_dir, "stats")
    output_train_dir = os.path.join(output_path, "train/")
    output_valid_dir = os.path.join(output_path, "valid/")

    # Make sure we have a clean worker space for Dask
    if os.path.isdir(dask_workdir):
        shutil.rmtree(dask_workdir)
    os.makedirs(dask_workdir)

    # Make sure we have a clean stats space for Dask
    if os.path.isdir(stats_path):
        shutil.rmtree(stats_path)
    os.mkdir(stats_path)

    # Make sure we have a clean output path
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    os.mkdir(output_train_dir)
    os.mkdir(output_valid_dir)

    # Get train/valid files
    train_paths = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
    ]
    n_files = int(len(train_paths) * 0.9)
    valid_paths = train_paths[n_files:]
    train_paths = train_paths[:n_files]

    # Force dtypes for HugeCTR usage
    dict_dtypes = {}
    for col in cats:
        dict_dtypes[col] = np.int64
    for col in conts:
        dict_dtypes[col] = np.float32
    for col in labels:
        dict_dtypes[col] = np.float32

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)

    # Check if any device memory is already occupied
    for dev in devices.split(","):
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

    # Setup dask cluster and perform ETL
    with managed_client(dask_workdir, devices, device_limit, protocol) as client:
        # Setup RMM pool
        if device_pool_frac > 0.01:
            setup_rmm_pool(client, device_pool_size)

        # Define Dask NVTabular "Workflow"
        cont_features = conts >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()

        cat_features = cats >> ops.Categorify(out_path=stats_path, max_size=10000000)

        workflow = Workflow(cat_features + cont_features + labels, client=client)

        train_dataset = Dataset(train_paths, engine="parquet", part_size=part_size)
        valid_dataset = Dataset(valid_paths, engine="parquet", part_size=part_size)

        workflow.fit(train_dataset)

        workflow.transform(train_dataset).to_parquet(
            output_path=output_train_dir,
            shuffle=nvt_io.Shuffle.PER_WORKER,
            dtypes=dict_dtypes,
            cats=cats,
            conts=conts,
            labels=labels,
            out_files_per_proc=out_files_per_proc,
        )
        workflow.transform(valid_dataset).to_parquet(
            output_path=output_valid_dir,
            shuffle=nvt_io.Shuffle.PER_WORKER,
            dtypes=dict_dtypes,
            cats=cats,
            conts=conts,
            labels=labels,
            out_files_per_proc=out_files_per_proc,
        )

        workflow.save(os.path.join(output_path, "workflow"))

        return workflow
