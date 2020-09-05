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

import argparse
import os
import shutil
import time

from dask.distributed import Client, performance_report
from dask_cuda import LocalCUDACluster
import rmm

from nvtabular import Dataset, Workflow
from nvtabular import io as nvt_io
from nvtabular import ops as ops
from nvtabular.utils import device_mem_size


def setup_rmm_pool(client, pool_size):
    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=pool_size)
    return None


def main(args):

    # Input
    data_path = args.data_path
    freq_limit = args.freq_limit
    out_files_per_proc = args.files_per_proc
    high_card_columns = args.high_card.split(",")
    dashboard_port = args.dashboard_port
    if args.protocol == "ucx":
        UCX_TLS = os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc,sockcm")
        os.environ["UCX_TLS"] = UCX_TLS

    # Cleanup output directory
    BASE_DIR = args.out_path
    dask_workdir = os.path.join(BASE_DIR, "workdir")
    output_path = os.path.join(BASE_DIR, "output")
    stats_path = os.path.join(BASE_DIR, "stats")
    if not os.path.isdir(BASE_DIR):
        os.mkdir(BASE_DIR)
    for dir_path in (dask_workdir, output_path, stats_path):
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        os.mkdir(dir_path)

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

    # Setup LocalCUDACluster
    if args.protocol == "tcp":
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devs,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
            dashboard_address=":" + dashboard_port,
        )
    else:
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devs,
            enable_nvlink=True,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
            dashboard_address=":" + dashboard_port,
        )
    client = Client(cluster)

    # Setup RMM pool
    if not args.no_rmm_pool:
        setup_rmm_pool(client, device_pool_size)

    # Define Dask NVTabular "Workflow"
    processor = Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, client=client
    )
    processor.add_feature([ops.FillMissing(), ops.Clip(min_value=0), ops.LogOp()])
    processor.add_preprocess(
        ops.Categorify(
            out_path=stats_path,
            tree_width=tree_width,
            cat_cache=cat_cache,
            freq_threshold=freq_limit,
            on_host=args.cat_on_host,
        )
    )
    processor.finalize()

    dataset = Dataset(data_path, "parquet", part_size=part_size)

    # Execute the dask graph
    runtime = time.time()
    if args.profile is not None:
        with performance_report(filename=args.profile):
            processor.apply(
                dataset,
                shuffle=nvt_io.Shuffle.PER_WORKER
                if args.worker_shuffle
                else nvt_io.Shuffle.PER_PARTITION,
                out_files_per_proc=out_files_per_proc,
                output_path=output_path,
            )
    else:
        processor.apply(
            dataset,
            shuffle=nvt_io.Shuffle.PER_WORKER
            if args.worker_shuffle
            else nvt_io.Shuffle.PER_PARTITION,
            out_files_per_proc=out_files_per_proc,
            output_path=output_path,
        )
    runtime = time.time() - runtime

    print("\nDask-NVTabular DLRM/Criteo benchmark")
    print("--------------------------------------")
    print(f"partition size     | {part_size}")
    print(f"protocol           | {args.protocol}")
    print(f"device(s)          | {args.devs}")
    print(f"rmm-pool           | {(not args.no_rmm_pool)}")
    print(f"out_files_per_proc | {args.files_per_proc}")
    print(f"worker-shuffle     | {args.worker_shuffle}")
    print("======================================")
    print(f"Runtime[s]         | {runtime}")
    print("======================================\n")

    client.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Merge (dask/cudf) on LocalCUDACluster benchmark")
    parser.add_argument(
        "-d", "--devs", default="0,1", type=str, help='GPU devices to use (default "0, 1").'
    )
    parser.add_argument(
        "-p",
        "--protocol",
        choices=["tcp", "ucx"],
        default="tcp",
        type=str,
        help="The communication protocol to use.",
    )
    parser.add_argument("--no-rmm-pool", action="store_true", help="Disable the RMM memory pool")
    parser.add_argument(
        "--profile",
        metavar="PATH",
        default=None,
        type=str,
        help="Write dask profile report (E.g. dask-report.html)",
    )
    parser.add_argument(
        "--dashboard-port",
        default="8787",
        type=str,
        help="Specify diagnostics-dashboard address (Default 8787).",
    )
    parser.add_argument("--data-path", type=str, help="Raw dataset path.")
    parser.add_argument("--out-path", type=str, help="Root output path.")
    parser.add_argument(
        "-s",
        "--files-per-proc",
        default=8,
        type=int,
        help="Number of files each worker process will write.",
    )
    parser.add_argument(
        "--part-mem-frac",
        default=0.125,
        type=float,
        help="Fraction of device memory for each partition",
    )
    parser.add_argument(
        "-f", "--freq-limit", default=0, type=int, help="Frequency limit on cat encodings."
    )
    parser.add_argument(
        "--device-limit-frac",
        default=0.8,
        type=float,
        help="Fractional device-memory limit (per worker).",
    )
    parser.add_argument(
        "--device-pool-frac", default=0.9, type=float, help="Fractional rmm pool size (per worker)."
    )
    parser.add_argument(
        "--worker-shuffle", action="store_true", help="Perform followup shuffle on each worker."
    )
    parser.add_argument(
        "--cat-names", default=None, type=str, help="List of categorical column names."
    )
    parser.add_argument(
        "--cont-names", default=None, type=str, help="List of continuous column names."
    )
    parser.add_argument(
        "--cat-cache-high",
        choices=["device", "host", "disk"],
        default="host",
        type=str,
        help='Where to cache high-cardinality category (Ex "host").',
    )
    parser.add_argument(
        "--cat-cache-low",
        choices=["device", "host", "disk"],
        default="device",
        type=str,
        help='Where to cache low-cardinality category (Ex "device, host, disk").',
    )
    parser.add_argument(
        "--cat-on-host",
        action="store_true",
        help="Whether to move categorical data to host between tasks.",
    )
    parser.add_argument(
        "--high-card",
        default="C20,C1,C22,C10",
        type=str,
        help="Specify a list of high-cardinality columns.  The tree-width "
        "and cat-cache options will apply to these columns only."
        '(Ex "C20,C1,C22,C10").',
    )
    parser.add_argument(
        "--tree-width",
        default=8,
        type=int,
        help="Tree width for GroupbyStatistics operations on high-cardinality "
        "columns (Default 4).",
    )
    args = parser.parse_args()
    args.n_workers = len(args.devs.split(","))
    return args


if __name__ == "__main__":
    main(parse_args())
