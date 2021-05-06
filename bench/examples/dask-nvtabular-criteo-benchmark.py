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
import os
import shutil
import time
import warnings

try:
    import boto3
except ImportError:
    boto3 = None

import rmm
from dask.distributed import Client, performance_report
from dask_cuda import LocalCUDACluster

try:
    from google.cloud import storage
except ImportError:
    storage = None

from nvtabular import Dataset, Workflow
from nvtabular import io as nvt_io
from nvtabular import ops
from nvtabular.utils import _pynvml_mem_size, device_mem_size, get_rmm_size


def setup_rmm_pool(client, pool_size):
    # Initialize an RMM pool allocator.
    # Note: RMM may require the pool size to be a multiple of 256.
    pool_size = get_rmm_size(pool_size)
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=pool_size)
    return None


def setup_dirs(base_dir, dask_workdir, output_path, stats_path):
    # GCP Storage
    if "gs://" in base_dir:
        # Check module is imported
        if storage is None:
            raise ImportError("google.cloud is not imported")
        # Get client and bucket
        storage_client = storage.Client()
        bucket_name = base_dir.split("/")[2]
        bucket = storage_client.bucket(bucket_name)
        # Delete all the objects within the directories
        for dir_path in (dask_workdir, output_path, stats_path):
            blobs = bucket.list_blobs(prefix=dir_path.split(bucket_name)[1][1:])
            for blob in blobs:
                blob.delete()

    # AWS Storage
    elif "s3://" in base_dir:
        # Check module is imported
        if boto3 is None:
            raise ImportError("boto3 is not imported")
        # Get client and bucket
        s3 = boto3.resource("s3")
        bucket_name = base_dir.split("/")[2]
        bucket = s3.Bucket(bucket_name)
        # Delete all the objects within the directories
        for dir_path in (dask_workdir, output_path, stats_path):
            bucket.objects.filter(Prefix=dir_path.split(bucket_name)[1][1:]).delete()

    # Local Storage
    else:
        if not os.path.isdir(base_dir):
            os.mkdir(base_dir)
        for dir_path in (dask_workdir, output_path, stats_path):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            os.mkdir(dir_path)


def main(args):
    """Multi-GPU Criteo/DLRM Preprocessing Benchmark

    This benchmark is designed to measure the time required to preprocess
    the Criteo (1TB) dataset for Facebook’s DLRM model.  The user must specify
    the path of the raw dataset (using the `--data-path` flag), as well as the
    output directory for all temporary/final data (using the `--out-path` flag)

    Example Usage
    -------------

    python dask-nvtabular-criteo-benchmark.py
                        --data-path /path/to/criteo_parquet --out-path /out/dir/`


    Dataset Requirements (Parquet)
    ------------------------------

    This benchmark is designed with a parquet-formatted dataset in mind.
    While a CSV-formatted dataset can be processed by NVTabular, converting
    to parquet will yield significantly better performance.  To convert your
    dataset, try using the `optimize_criteo.ipynb` notebook (also located
    in `NVTabular/examples/`)

    For a detailed parameter overview see `NVTabular/examples/MultiGPUBench.md`
    """

    # Input
    data_path = args.data_path[:-1] if args.data_path[-1] == "/" else args.data_path
    freq_limit = args.freq_limit
    out_files_per_proc = args.out_files_per_proc
    high_card_columns = args.high_cards.split(",")
    dashboard_port = args.dashboard_port
    if args.protocol == "ucx":
        UCX_TLS = os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc,sockcm")
        os.environ["UCX_TLS"] = UCX_TLS

    # Cleanup output directory
    base_dir = args.out_path[:-1] if args.out_path[-1] == "/" else args.out_path
    dask_workdir = os.path.join(base_dir, "workdir")
    output_path = os.path.join(base_dir, "output")
    stats_path = os.path.join(base_dir, "stats")
    setup_dirs(base_dir, dask_workdir, output_path, stats_path)

    # Use Criteo dataset by default (for now)
    cont_names = (
        args.cont_names.split(",") if args.cont_names else ["I" + str(x) for x in range(1, 14)]
    )
    cat_names = (
        args.cat_names.split(",") if args.cat_names else ["C" + str(x) for x in range(1, 27)]
    )
    label_name = ["label"]

    # Specify Categorify/GroupbyStatistics options
    tree_width = {}
    cat_cache = {}
    for col in cat_names:
        if col in high_card_columns:
            tree_width[col] = args.tree_width
            cat_cache[col] = args.cat_cache_high
        else:
            tree_width[col] = 1
            cat_cache[col] = args.cat_cache_low

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    device_limit = int(args.device_limit_frac * device_size)
    device_pool_size = int(args.device_pool_frac * device_size)
    part_size = int(args.part_mem_frac * device_size)

    # Parse shuffle option
    shuffle = None
    if args.shuffle == "PER_WORKER":
        shuffle = nvt_io.Shuffle.PER_WORKER
    elif args.shuffle == "PER_PARTITION":
        shuffle = nvt_io.Shuffle.PER_PARTITION

    # Check if any device memory is already occupied
    for dev in args.devices.split(","):
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

    # Setup LocalCUDACluster
    if args.protocol == "tcp":
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devices,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
            dashboard_address=":" + dashboard_port,
        )
    else:
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devices,
            enable_nvlink=True,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
            dashboard_address=":" + dashboard_port,
        )
    client = Client(cluster)

    # Setup RMM pool
    if args.device_pool_frac > 0.01:
        setup_rmm_pool(client, device_pool_size)

    # Define Dask NVTabular "Workflow"
    if args.normalize:
        cont_features = cont_names >> ops.FillMissing() >> ops.Normalize()
    else:
        cont_features = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()

    cat_features = cat_names >> ops.Categorify(
        out_path=stats_path,
        tree_width=tree_width,
        cat_cache=cat_cache,
        freq_threshold=freq_limit,
        search_sorted=not freq_limit,
        on_host=not args.cats_on_device,
    )
    processor = Workflow(cat_features + cont_features + label_name, client=client)

    dataset = Dataset(data_path, "parquet", part_size=part_size)

    # Execute the dask graph
    runtime = time.time()

    processor.fit(dataset)

    if args.profile is not None:
        with performance_report(filename=args.profile):
            processor.transform(dataset).to_parquet(
                output_path=output_path,
                num_threads=args.num_io_threads,
                shuffle=shuffle,
                out_files_per_proc=out_files_per_proc,
            )
    else:
        processor.transform(dataset).to_parquet(
            output_path=output_path,
            num_threads=args.num_io_threads,
            shuffle=shuffle,
            out_files_per_proc=out_files_per_proc,
        )
    runtime = time.time() - runtime

    print("\nDask-NVTabular DLRM/Criteo benchmark")
    print("--------------------------------------")
    print(f"partition size     | {part_size}")
    print(f"protocol           | {args.protocol}")
    print(f"device(s)          | {args.devices}")
    print(f"rmm-pool-frac      | {(args.device_pool_frac)}")
    print(f"out-files-per-proc | {args.out_files_per_proc}")
    print(f"num_io_threads     | {args.num_io_threads}")
    print(f"shuffle            | {args.shuffle}")
    print(f"cats-on-device     | {args.cats_on_device}")
    print("======================================")
    print(f"Runtime[s]         | {runtime}")
    print("======================================\n")

    client.close()


def parse_args():
    parser = argparse.ArgumentParser(description=("Multi-GPU Criteo/DLRM Preprocessing Benchmark"))

    #
    # System Options
    #

    parser.add_argument("--data-path", type=str, help="Input dataset path (Required)")
    parser.add_argument("--out-path", type=str, help="Directory path to write output (Required)")
    parser.add_argument(
        "-d",
        "--devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
        type=str,
        help='Comma-separated list of visible devices (e.g. "0,1,2,3"). '
        "The number of visible devices dictates the number of Dask workers (GPU processes) "
        "The CUDA_VISIBLE_DEVICES environment variable will be used by default",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="Communication protocol to use (Default 'tcp')",
    )
    parser.add_argument(
        "--device-limit-frac",
        default=0.8,
        type=float,
        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). "
        "The worker will try to spill data to host memory beyond this limit",
    )
    parser.add_argument(
        "--device-pool-frac",
        default=0.9,
        type=float,
        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.9). "
        "If 0 is specified, the RMM pool will be disabled",
    )
    parser.add_argument(
        "--num-io-threads",
        default=0,
        type=int,
        help="Number of threads to use when writing output data (Default 0). "
        "If 0 is specified, multi-threading will not be used for IO.",
    )

    #
    # Data-Decomposition Parameters
    #

    parser.add_argument(
        "--part-mem-frac",
        default=0.125,
        type=float,
        help="Maximum size desired for dataset partitions as a fraction "
        "of GPU capacity (Default 0.125)",
    )
    parser.add_argument(
        "--out-files-per-proc",
        default=8,
        type=int,
        help="Number of output files to write on each worker (Default 8)",
    )

    #
    # Preprocessing Options
    #

    parser.add_argument(
        "-f",
        "--freq-limit",
        default=0,
        type=int,
        help="Frequency limit for categorical encoding (Default 0)",
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        choices=["PER_WORKER", "PER_PARTITION", "NONE"],
        default="PER_PARTITION",
        help="Shuffle algorithm to use when writing output data to disk (Default PER_PARTITION)",
    )
    parser.add_argument(
        "--cat-names", default=None, type=str, help="List of categorical column names (Optional)"
    )
    parser.add_argument(
        "--cont-names", default=None, type=str, help="List of continuous column names (Optional)"
    )
    parser.add_argument("--normalize", action="store_true", help="Normalize continuous features.")

    #
    # Algorithm Options
    #

    parser.add_argument(
        "--cats-on-device",
        action="store_true",
        help="Keep intermediate GroupbyStatistics results in device memory between tasks."
        "This is recommended when the total device memory is sufficiently large.",
    )
    parser.add_argument(
        "--high-cards",
        default="C20,C1,C22,C10",
        type=str,
        help="Specify a list of high-cardinality columns.  The tree-width "
        "and cat-cache options will apply to these columns only."
        '(Default "C20,C1,C22,C10")',
    )
    parser.add_argument(
        "--tree-width",
        default=8,
        type=int,
        help="Tree width for GroupbyStatistics operations on high-cardinality "
        "columns (Default 8)",
    )
    parser.add_argument(
        "--cat-cache-high",
        choices=["device", "host", "disk"],
        default="host",
        type=str,
        help='Where to cache high-cardinality category (Default "host")',
    )
    parser.add_argument(
        "--cat-cache-low",
        choices=["device", "host", "disk"],
        default="device",
        type=str,
        help='Where to cache low-cardinality category (Default "device")',
    )

    #
    # Diagnostics Options
    #

    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Specify a file path to export a Dask profile report (E.g. dask-report.html)."
        "If this option is excluded from the command, not profile will be exported",
    )
    parser.add_argument(
        "--dashboard-port",
        default="8787",
        type=str,
        help="Specify the desired port of Dask's diagnostics-dashboard (Default `3787`). "
        "The dashboard will be hosted at http://<IP>:<PORT>/status",
    )
    args = parser.parse_args()
    args.n_workers = len(args.devices.split(","))
    return args


if __name__ == "__main__":
    main(parse_args())
