# Note: Be sure to clean up output and dask work-space before running test

import argparse
import os
import time

import cudf
import rmm

import nvtabular.ops as ops
from dask.distributed import Client, performance_report
from dask_cuda import LocalCUDACluster
from nvtabular import DaskDataset, Workflow


def setup_rmm_pool(client, pool_size):
    client.run(cudf.set_allocator, pool=True, initial_pool_size=pool_size, allocator="default")
    return None


def main(args):

    # Input
    data_path = args.data_path
    out_path = args.out_path
    freq_limit = args.freq_limit
    nsplits = args.splits
    if args.protocol == "ucx":
        os.environ["UCX_TLS"] = "tcp,cuda_copy,cuda_ipc,sockcm"

    # Use Criteo dataset by default (for now)
    cont_names = (
        args.cont_names.split(",") if args.cont_names else ["I" + str(x) for x in range(1, 14)]
    )
    cat_names = (
        args.cat_names.split(",") if args.cat_names else ["C" + str(x) for x in range(1, 27)]
    )
    label_name = ["label"]

    if args.cat_splits:
        split_out = {name: int(s) for name, s in zip(cat_names, args.cat_splits.split(","))}
    else:
        split_out = {col: 1 for col in cat_names}
        if args.cat_names is None:
            # Using Criteo... Use more hash partitions for
            # known high-cardinality columns
            if args.n_workers >= 4:
                split_out["C20"] = 8
                split_out["C1"] = 8
                split_out["C22"] = 4
                split_out["C10"] = 4
                split_out["C21"] = 2
                split_out["C11"] = 2
                split_out["C23"] = 2
                split_out["C12"] = 2
            else:
                split_out["C20"] = 8
                split_out["C1"] = 8
                split_out["C22"] = 4
                split_out["C10"] = 4
                split_out["C21"] = 2
                split_out["C11"] = 2
                split_out["C23"] = 2
                split_out["C12"] = 2

    # Specify categorical caching location
    cat_cache = None
    if args.cat_cache:
        cat_cache = args.cat_cache.split(",")
        if len(cat_cache) == 1:
            cat_cache = cat_cache[0]
        else:
            # If user is specifying a list of options,
            # they must specify an option for every cat column
            assert len(cat_names) == len(cat_cache)
    if isinstance(cat_cache, str):
        cat_cache = {col: cat_cache for col in cat_names}
    elif isinstance(cat_cache, list):
        cat_cache = {name: c for name, c in zip(cat_names, cat_cache)}
    else:
        # Criteo/DLRM Defaults
        cat_cache = {col: "device" for col in cat_names}
        if args.cat_names is None:
            cat_cache["C20"] = "host"
            cat_cache["C1"] = "host"
            # Only need to cache the largest two on a dgx-2
            if args.n_workers < 16:
                cat_cache["C22"] = "host"
                cat_cache["C10"] = "host"

    # Use total device size to calculate args.device_limit_frac
    device_size = rmm.get_info().total
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
            local_directory=args.dask_workspace,
            dashboard_address=":3787",
        )
    else:
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devs,
            enable_nvlink=True,
            device_memory_limit=device_limit,
            local_directory=args.dask_workspace,
            dashboard_address=":3787",
        )
    client = Client(cluster)

    # Setup RMM pool
    if not args.no_rmm_pool:
        setup_rmm_pool(client, device_pool_size)

    # Define Dask NVTabular "Workflow"
    processor = Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, client=client,
    )
    processor.add_feature([ops.ZeroFill(), ops.LogOp()])
    processor.add_preprocess(
        ops.Categorify(
            out_path=out_path, split_out=split_out, cat_cache=cat_cache, freq_threshold=freq_limit,
        )
    )
    processor.finalize()

    dataset = DaskDataset(data_path, "parquet", part_size=part_size)

    # Execute the dask graph
    runtime = time.time()
    if args.profile is not None:
        with performance_report(filename=args.profile):
            processor.apply(
                dataset,
                shuffle="full" if args.worker_shuffle else "partial",
                nsplits=nsplits,
                output_path=out_path,
            )
    else:
        processor.apply(
            dataset,
            shuffle="full" if args.worker_shuffle else "partial",
            nsplits=nsplits,
            output_path=out_path,
        )
    runtime = time.time() - runtime

    print("\nDask-NVTabular DLRM/Criteo benchmark")
    print("--------------------------------------")
    print(f"partition size  | {part_size}")
    print(f"protocol        | {args.protocol}")
    print(f"device(s)       | {args.devs}")
    print(f"rmm-pool        | {(not args.no_rmm_pool)}")
    print(f"nsplits         | {args.splits}")
    print(f"worker-shuffle  | {args.worker_shuffle}")
    print("======================================")
    print(f"Runtime[s]      | {runtime}")
    print("======================================\n")

    client.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Merge (dask/cudf) on LocalCUDACluster benchmark")
    parser.add_argument(
        "-d",
        "--devs",
        default="0,1,2,3",
        type=str,
        help='GPU devices to use (default "0, 1, 2, 3").',
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
        "--data-path", type=str, help="Raw dataset path.",
    )
    parser.add_argument("--out-path", type=str, help="Root output path.")
    parser.add_argument(
        "--dask-workspace", default=None, type=str, help="Dask workspace path.",
    )
    parser.add_argument(
        "-s", "--splits", default=24, type=int, help="Number of splits to shuffle each partition"
    )
    parser.add_argument(
        "--part-mem-frac",
        default=0.162,
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
        "--device-pool-frac",
        default=0.8,
        type=float,
        help="Fractional rmm pool size (per worker).",
    )
    parser.add_argument(
        "--worker-shuffle", action="store_true", help="Perform followup shuffle on each worker."
    )
    parser.add_argument(
        "--cat-names", default=None, type=str, help="List of categorical column names."
    )
    parser.add_argument(
        "--cat-cache",
        default=None,
        type=str,
        help='Where to cache each category (Ex "device, host, disk").',
    )
    parser.add_argument(
        "--cat-splits",
        default=None,
        type=str,
        help='How many splits to use for each category (Ex "8, 4, 2, 1").',
    )
    parser.add_argument(
        "--cont-names", default=None, type=str, help="List of continuous column names."
    )
    args = parser.parse_args()
    args.n_workers = len(args.devs.split(","))
    return args


if __name__ == "__main__":
    main(parse_args())
