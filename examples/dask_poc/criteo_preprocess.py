# Note: Be sure to clean up output and dask work-space before running test

import os
import glob
import time
import fsspec
import argparse
from io import BytesIO

import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe.core import _concat
from fsspec.core import get_fs_token_paths
import pyarrow.parquet as pq
from pyarrow.compat import guid
from operator import getitem

import numpy as np
import pandas as pd

import cupy
import cudf
import dask_cudf
from cudf.io.parquet import ParquetWriter

from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait, performance_report, get_worker
from dask.utils import parse_bytes, natural_sort_key
from dask.dataframe.utils import group_split_dispatch

""" Caching Utilities
"""


class CategoryCache:
    def __init__(self):
        self.cat_cache = {}
        self.pq_writer_cache = {}

    def __del__(self):
        for path, (pw, fpath) in self.pq_writer_cache.items():
            pw.close()

    def _get_pq_writer(self, prefix, s, mem):
        pw, fil = self.pq_writer_cache.get(prefix, (None, None))
        if pw is None:
            if mem:
                fil = BytesIO()
                pw = ParquetWriter(fil, compression=None)
                self.pq_writer_cache[prefix] = (pw, fil)
            else:
                outfile_id = guid() + ".parquet"
                full_path = ".".join([prefix, outfile_id])
                pw = ParquetWriter(full_path, compression=None)
                self.pq_writer_cache[prefix] = (pw, full_path)
        return pw

    def _get_encodings(self, col, path, cache=False):
        table = self.cat_cache.get(col, None)
        if table and not isinstance(table, cudf.DataFrame):
            df = cudf.io.read_parquet(table, index=False, columns=[col])
            df.index.name = "labels"
            df.reset_index(drop=False, inplace=True)
            return df

        if table is None:
            if cache:
                table = cudf.io.read_parquet(path, index=False, columns=[col])
            else:
                with open(path, "rb") as f:
                    self.cat_cache[col] = BytesIO(f.read())
                table = cudf.io.read_parquet(self.cat_cache[col], index=False, columns=[col])
            table.index.name = "labels"
            table.reset_index(drop=False, inplace=True)
            if cache:
                self.cat_cache[col] = table.copy(deep=False)

        return table


def _cache():
    worker = get_worker()
    if not hasattr(worker, "cats_cache"):
        worker.cats_cache = CategoryCache()
    return worker.cats_cache


def _clean_cache():
    worker = get_worker()
    if hasattr(worker, "cats_cache"):
        del worker.cats_cache
    return


""" Helper Functions
"""


def setup_rmm_pool(client, pool_size="24GB"):
    client.run(
        cudf.set_allocator, pool=True, initial_pool_size=parse_bytes(pool_size), allocator="default"
    )
    return None


def get_dataset_parts(data_path, fs, row_groups_per_part):
    parts = []
    dataset = pq.ParquetDataset(data_path, filesystem=fs)
    if dataset.metadata:
        fpath_last = None
        rgi = 0
        rg_list = []
        for rg in range(dataset.metadata.num_row_groups):

            fpath = dataset.metadata.row_group(rg).column(0).file_path

            if fpath_last and fpath_last != fpath:
                rgi = 0
                full_path = fs.sep.join([data_path, fpath_last])
                parts.append(tuple([full_path, rg_list]))
                rg_list = []
            elif len(rg_list) >= row_groups_per_part:
                full_path = fs.sep.join([data_path, fpath_last])
                parts.append(tuple([full_path, rg_list]))
                rg_list = []

            if fpath is None:
                print("ERROR - File path missing from metadata!")

            fpath_last = fpath
            rg_list.append(rgi)
            rgi += 1
        if rg_list:
            full_path = fs.sep.join([data_path, fpath_last])
            parts.append(tuple([full_path, rg_list]))
    else:
        print("WARNING - Must have metadata file to split by row-group chunks!")
        for piece in dataset.pieces:
            parts.append(tuple([piece.path, None]))
    return parts


def _finish_labels(paths, cat_names):
    return {name: paths[i] for i, name in enumerate(cat_names)}


def _combine_unique(dfs, col, freq_limit):
    ignore_index = True
    gb = _concat(dfs, ignore_index).groupby(col, dropna=False).count()
    gb.reset_index(drop=False, inplace=True)
    # Can filter out low-frequecy items here
    if freq_limit:
        gb = gb[gb["_count"] >= freq_limit]
    return gb[[col]]


def _write_unique(dfs, base_path, col, fs, kwargs):
    ignore_index = True
    df = _concat(dfs, ignore_index,)
    rel_path = "unique.%s.parquet" % (col)
    path = fs.sep.join([base_path, rel_path])
    if len(df):
        # Make sure first category is Null
        df = df.sort_values(col, na_position="first")
        if not df[col]._column.has_nulls:
            df = cudf.DataFrame({col: _concat([cudf.Series([None]), df[col]], ignore_index,)})
        df.to_parquet(path, write_index=False, compression=None, **kwargs)
    else:
        path = None
    del df
    return path


def _read_and_process(part, columns, split_out, kwargs):

    # Read dataset part
    path, row_groups = part
    if row_groups:
        gdf = cudf.io.read_parquet(
            path,
            row_group=row_groups[0],
            row_group_count=len(row_groups),
            index=False,
            columns=columns,
            **kwargs,
        )
    else:
        gdf = cudf.io.read_parquet(path, index=False, columns=columns, **kwargs)
    gdf["_count"] = cupy.ones(len(gdf), dtype="int32")

    # CATEGORIFY
    # Write a hash-partitioned dataset
    # (to expose concurrency in follow-up pass)
    output = {}
    k = 0
    for i, col in enumerate(columns):
        gb = gdf[[col, "_count"]].groupby(col, dropna=False).count()
        gb.reset_index(drop=False, inplace=True)
        for j, split in enumerate(gb.partition_by_hash([col], split_out[col], keep_index=False)):
            output[k] = split
            k += 1
        gdf.drop(columns=[col], inplace=True)
        del gb
    del gdf
    return output


def _read_and_apply_ops(
    part, cat_cols, cont_cols, cat_labels, nsplits, processed_path, fs, gpu_cache, mem, full, kwargs
):

    # Read dataset part
    path, row_groups = part
    if row_groups:
        gdf = cudf.io.read_parquet(
            path, row_group=row_groups[0], row_group_count=len(row_groups), index=False, **kwargs
        )
    else:
        gdf = cudf.io.read_parquet(path, index=False, **kwargs)

    def _zero_fill(gdf: cudf.DataFrame, target_columns: list):
        cont_names = target_columns
        if not cont_names:
            return gdf
        z_gdf = gdf[cont_names].fillna(0)
        z_gdf[z_gdf < 0] = 0
        for col in cont_names:
            gdf[col] = z_gdf[col]
        return gdf

    def _log_op(gdf: cudf.DataFrame, target_columns: list):
        cont_names = target_columns
        if not cont_names:
            return gdf
        new_gdf = np.log(gdf[cont_names].astype(np.float32) + 1)
        for col in cont_names:
            gdf[col] = new_gdf[col]
        return gdf

    # Continuous-Variable Operations
    if cont_cols:
        gdf = _zero_fill(gdf, cont_cols)
        gdf = _log_op(gdf, cont_cols)

    # Categorical Encodings
    def _encode(vals, encodings, col, gpu_cache, na_sentinel=-1):
        if encodings[col]:
            value = _cache()._get_encodings(col, encodings[col], cache=gpu_cache[col])
        else:
            value = cudf.DataFrame({col: [None]})
            value[col] = value[col].astype(vals.dtype)
            value.index.name = "labels"
            value.reset_index(drop=False, inplace=True)

        if full:
            # Use `searchsorted` if we are using a "full" encoding
            labels = value[col].searchsorted(vals, side="left", na_position="first")
            labels[labels >= len(value[col])] = na_sentinel
            return labels
        else:
            codes = cudf.DataFrame({col: vals.copy(), "order": cupy.arange(len(vals)),})
            codes = codes.merge(value, on=col, how="left").sort_values("order")["labels"]
            codes.fillna(na_sentinel, inplace=True)
            return codes.values

    for col in cat_cols:
        gdf[col] = _encode(gdf[col]._column, cat_labels, col, gpu_cache, na_sentinel=0)

    # Split gdf into random splits
    gdf_size = len(gdf)
    if mem:
        # Dont need a real sort if we are doing in memory later
        ind = cupy.random.choice(cupy.arange(nsplits, dtype="int8"), gdf_size)
        result = group_split_dispatch(gdf, ind, nsplits, ignore_index=True)
        del ind
        del gdf
        # Write each split to a separate file
        for s, df in result.items():
            prefix = fs.sep.join([processed_path, "split." + str(s)])
            pw = _cache()._get_pq_writer(prefix, s, mem=mem)
            pw.write_table(df)
    else:
        # We should do a real sort here
        sort_key = "__sort_index__"
        arr = cupy.arange(gdf_size)
        cupy.random.shuffle(arr)
        gdf[sort_key] = arr
        gdf = gdf.sort_values(sort_key)
        gdf.drop(columns=[sort_key], inplace=True)
        splits = list(range(0, gdf_size, int(gdf_size / nsplits)))
        splits[-1] = gdf_size
        # Write each split to a separate file
        for s in range(0, len(splits) - 1):
            prefix = fs.sep.join([processed_path, "split." + str(s)])
            pw = _cache()._get_pq_writer(prefix, s, mem=mem)
            pw.write_table(gdf.iloc[splits[s] : splits[s + 1]])

    return gdf_size


def _finish_write(parts):
    return parts


def _worker_shuffle(processed_path, fs):
    paths = []
    for path, (pw, bio) in _cache().pq_writer_cache.items():
        pw.close()

        gdf = cudf.io.read_parquet(bio, index=False,)
        bio.close()

        sort_key = "__sort_index__"
        arr = cupy.arange(len(gdf))
        cupy.random.shuffle(arr)
        gdf[sort_key] = arr
        gdf = gdf.sort_values(sort_key)
        gdf.drop(columns=[sort_key], inplace=True)

        rel_path = "shuffled.%s.parquet" % (guid())
        full_path = fs.sep.join([processed_path, rel_path])
        gdf.to_parquet(
            full_path, compression=None, index=False,
        )
        paths.append(full_path)

    return paths


""" Main
"""


def main(args):

    # Input
    data_path = args.data_path
    out_path = args.out_path
    freq_limit = args.freq_limit
    nsplits = args.splits
    row_groups_per_part = args.row_groups
    if args.protocol == "ucx":
        os.environ["UCX_TLS"] = "tcp,cuda_copy,cuda_ipc,sockcm"

    # Use Criteo dataset by default (for now)
    cont_names = (
        args.cont_names.split(",") if args.cont_names else ["I" + str(x) for x in range(1, 14)]
    )
    cat_names = (
        args.cat_names.split(",") if args.cat_names else ["C" + str(x) for x in range(1, 27)]
    )

    if args.cat_splits:
        split_out = {name: int(s) for name, s in zip(cat_names, args.cat_splits.split(","))}
    else:
        split_out = {col: 1 for col in cat_names}
        if args.cat_names is None:
            # Using Criteo... Use more hash partitions for
            # known high-cardinality columns
            split_out["C20"] = 8
            split_out["C1"] = 8
            split_out["C22"] = 4
            split_out["C10"] = 4
            split_out["C21"] = 2
            split_out["C11"] = 2
            split_out["C23"] = 2
            split_out["C12"] = 2

    if args.cat_cache:
        gpu_cache = {name: bool(c) for name, c in zip(cat_names, args.cat_cache.split(","))}
    else:
        # Chose which cat_coluns to cache directly in device memory
        gpu_cache = {col: True for col in cat_names}
        if args.cat_names is None:
            gpu_cache["C20"] = False
            gpu_cache["C1"] = False
            # Only need to cache the largest two on a dgx-2
            if args.n_workers < 16:
                gpu_cache["C22"] = False
                gpu_cache["C10"] = False

    # Setup LocalCUDACluster
    if args.protocol == "tcp":
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devs,
            device_memory_limit=parse_bytes(args.device_limit),
            memory_limit=parse_bytes(args.host_limit),
            local_directory=args.dask_workspace,
        )
    else:
        cluster = LocalCUDACluster(
            protocol=args.protocol,
            n_workers=args.n_workers,
            CUDA_VISIBLE_DEVICES=args.devs,
            enable_nvlink=True,
            device_memory_limit=parse_bytes(args.device_limit),
            memory_limit=parse_bytes(args.host_limit),
            local_directory=args.dask_workspace,
        )
    client = Client(cluster)

    # Setup RMM pool and caching
    if not args.no_rmm_pool:
        setup_rmm_pool(client, parse_bytes(args.device_limit))
    client.run(_cache)

    fs = get_fs_token_paths(data_path, mode="rb")[0]
    parts = get_dataset_parts(data_path, fs, row_groups_per_part)
    processed_path = out_path + fs.sep + "processed"
    fs.mkdirs(processed_path, exist_ok=True)

    # Debug - Take partial dataset
    if args.debug:
        parts = parts[:nsplits]

    # Build task graph to calculate categorical encodings
    dsk = {}
    token = tokenize(parts, out_path)
    read_name = "read-pq-" + token
    reread_name = "reread-pq-" + token
    split_name = "shuffle-split-" + token
    split_2_name = "shuffle-split-2-" + token
    combined_hash_name = "shuffle-combine-" + token
    combined_write_name = "shuffle-write-" + token
    finalize_labels_name = "finalize_encoding-" + token
    finalize_write_name = "finish-write-" + token
    for p, part in enumerate(parts):

        # Step 1: Read/process datset files
        read_key = (read_name, p)
        dsk[read_key] = (
            _read_and_process,
            part,
            cat_names,
            split_out,
            {},
        )

        # Step 2: Separate each categorical column and hash split
        k = 0
        for c, col in enumerate(cat_names):
            for s in range(split_out[col]):
                split_key = (split_name, p, c, s)
                dsk[split_key] = (getitem, (read_name, p), k)
                k += 1

    # Step 3: Combine each hash/column combo
    for c, col in enumerate(cat_names):
        for s in range(split_out[col]):
            combined_hash_key = (combined_hash_name, c, s)
            dsk[combined_hash_key] = (
                _combine_unique,
                [(split_name, p, c, s) for p in range(len(parts))],
                col,
                freq_limit,
            )

        # Step 4: Aggregate hash partitions and write to disk
        combined_write_key = (combined_write_name, c)
        dsk[combined_write_key] = (
            _write_unique,
            [(combined_hash_name, c, s) for s in range(split_out[col])],
            out_path,
            col,
            fs,
            {},
        )

    # Step 5: Finalize Labels
    dsk[finalize_labels_name] = (
        _finish_labels,
        [(combined_write_name, c) for c, col in enumerate(cat_names)],
        cat_names,
    )

    # Step 6: Re-read dataset and write transformed/shuffled dataset
    use_bytesio = args.worker_shuffle
    for p, part in enumerate(parts):
        reread_key = (reread_name, p)
        dsk[reread_key] = (
            _read_and_apply_ops,
            part,
            cat_names,
            cont_names,
            finalize_labels_name,
            nsplits,
            processed_path,
            fs,
            gpu_cache,
            use_bytesio,
            (freq_limit == 0),
            {},
        )

    # Step 7: Dummy task to tie everything together
    #         TODO: Write a "_metadata" file here
    dsk[finalize_write_name] = (
        _finish_write,
        [(reread_name, p) for p in range(len(parts))],
    )

    # Execute the dask graph
    runtime = time.time()
    if args.profile is not None:
        with performance_report(filename=args.profile):
            done = client.get(dsk, finalize_write_name)
            if args.worker_shuffle:
                worker_shuffle = client.run(_worker_shuffle, processed_path, fs)
            cleanup = client.run(_clean_cache)
    else:
        done = client.get(dsk, finalize_write_name)
        if args.worker_shuffle:
            worker_shuffle = client.run(_worker_shuffle, processed_path, fs)
        cleanup = client.run(_clean_cache)
    runtime = time.time() - runtime

    print("\nDask POC nvtabular benchmark")
    print("-------------------------------")
    print(f"row-group chunk | {args.row_groups}")
    print(f"protocol        | {args.protocol}")
    print(f"device(s)       | {args.devs}")
    print(f"rmm-pool        | {(not args.no_rmm_pool)}")
    print(f"nsplits         | {args.splits}")
    print(f"worker-shuffle  | {args.worker_shuffle}")
    print("===============================")
    print(f"Runtime[s]      | {runtime}")
    print("===============================\n")

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
        "-r", "--row-groups", default=48, type=int, help="Number of row-groups per partition"
    )
    parser.add_argument(
        "-f", "--freq-limit", default=0, type=int, help="Frequency limit on cat encodings."
    )
    parser.add_argument(
        "--device-limit", default="26GB", type=str, help="Device memory limit (per worker)."
    )
    parser.add_argument(
        "--host-limit", default="96GB", type=str, help="Host memory limit (per worker)."
    )
    parser.add_argument(
        "--worker-shuffle", action="store_true", help="Perform followup shuffle on each worker."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use fraction of dataset for debugging."
    )
    parser.add_argument(
        "--cat-names", default=None, type=str, help="List of categorical column names."
    )
    parser.add_argument(
        "--cat-cache",
        default=None,
        type=str,
        help='Whether to cache each category in device memory (Ex "1, 0, 0, 0").',
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
