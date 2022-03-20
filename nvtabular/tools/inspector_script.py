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

import argparse
import contextlib
import json
import os

import fsspec
import rmm
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular.tools.dataset_inspector as datains
from merlin.core.utils import device_mem_size, get_rmm_size
from merlin.io import Dataset


def setup_rmm_pool(client, device_pool_size):
    # Initialize an RMM pool allocator.
    # Note: RMM may require the pool size to be a multiple of 256.
    device_pool_size = get_rmm_size(device_pool_size)
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=device_pool_size)
    return None


@contextlib.contextmanager
def managed_client(devices, device_limit, protocol):
    client = Client(
        LocalCUDACluster(
            protocol=protocol,
            n_workers=len(devices.split(",")),
            enable_nvlink=(protocol == "ucx"),
            device_memory_limit=device_limit,
        )
    )
    try:
        yield client
    finally:
        client.shutdown()


def parse_args():
    """
    Use the inspector script indicating the config fil, path, format,
    gpus to use and (optional) the output file name

    python inspector_script.py -c config_file.json -d dataset_path -f parquet
                               -g "0,1,2,3,4" -o dataset_info.json
    """
    parser = argparse.ArgumentParser(description=("Dataset Inspect Tool"))
    # Config file
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="Dataset columns type (Required)",
    )
    # Dataset path
    parser.add_argument(
        "--data_path",
        default="0",
        type=str,
        help="Input dataset path (Required)",
    )

    # Number of GPUs to use
    parser.add_argument(
        "-d",
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
        "The number of visible devices dictates the number of Dask workers (GPU processes) "
        "The CUDA_VISIBLE_DEVICES environment variable will be used by default",
    )

    # Device limit
    parser.add_argument(
        "--device-limit-frac",
        default=0.8,
        type=float,
        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). "
        "The worker will try to spill data to host memory beyond this limit",
    )

    # RMM pool size
    parser.add_argument(
        "--device-pool-frac",
        default=0.9,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.9). "
        "If 0 is specified, the RMM pool will be disabled",
    )

    # Dataset format
    parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        type=str,
        help="Dataset format (Default 'parquet')",
    )

    # Partition size
    parser.add_argument(
        "--part-mem-frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )

    # Protocol
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="Communication protocol to use (Default 'tcp')",
    )

    # Output file name
    parser.add_argument(
        "-o",
        "--output_file",
        default="dataset_info.json",
        type=str,
        help="Output file name (Default 'dataset_info.json')",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Get device configuration
    device_size = device_mem_size(kind="total")
    device_limit = int(args.device_limit_frac * device_size)
    device_pool_size = int(args.device_pool_frac * device_size)
    part_size = int(args.part_mem_frac * device_size)

    # Get dataset columns
    with fsspec.open(args.config_file) as f:
        config = json.load(f)

    # Create Dataset
    dataset = Dataset(args.data_path, engine=args.format, part_size=part_size)

    # Call Inspector
    with managed_client(args.devices, device_limit, args.protocol) as client:
        setup_rmm_pool(client, device_pool_size)
        a = datains.DatasetInspector(client)
        a.inspect(dataset, config, args.output_file)


if __name__ == "__main__":
    main(parse_args())
