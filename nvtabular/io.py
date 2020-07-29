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

import collections
import functools
import json
import logging
import os
import queue
import threading
import warnings
from collections import defaultdict
from io import BytesIO
from uuid import uuid4

import cudf
import cupy as cp
import dask
import dask_cudf
import numba.cuda as cuda
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from cudf._lib.nvtx import annotate
from cudf.io.parquet import ParquetWriter as pwriter
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.dataframe.io.parquet.utils import _analyze_paths
from dask.dataframe.utils import group_split_dispatch
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.utils import natural_sort_key, parse_bytes
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path

from nvtabular.worker import clean_worker_cache, get_worker_cache

LOG = logging.getLogger("nvtabular")


#
# Helper Function definitions
#


def _allowable_batch_size(gpu_memory_frac, row_size):
    free_mem = device_mem_size(kind="free")
    gpu_memory = free_mem * gpu_memory_frac
    return max(int(gpu_memory / row_size), 1)


def _shuffle_gdf(gdf, gdf_size=None):
    """ Shuffles a cudf dataframe, returning a new dataframe with randomly
    ordered rows """
    gdf_size = gdf_size or len(gdf)
    arr = cp.arange(gdf_size)
    cp.random.shuffle(arr)
    return gdf.iloc[arr]


def device_mem_size(kind="total"):
    if kind not in ["free", "total"]:
        raise ValueError("{0} not a supported option for device_mem_size.".format(kind))
    try:
        if kind == "free":
            return int(cuda.current_context().get_memory_info()[0])
        else:
            return int(cuda.current_context().get_memory_info()[1])
    except NotImplementedError:
        import pynvml

        pynvml.nvmlInit()
        if kind == "free":
            warnings.warn("get_memory_info is not supported. Using total device memory from NVML.")
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).total)
        pynvml.nvmlShutdown()
    return size


def guid():
    """ Simple utility function to get random hex string
    """
    return uuid4().hex


def _merge_general_metadata(meta_list):
    """ Combine list of "general" metadata dicts into
        a single dict
    """
    if not meta_list:
        return {}
    meta = None
    for md in meta_list:
        if meta:
            meta["data_paths"] += md["data_paths"]
            meta["file_stats"] += md["file_stats"]
        else:
            meta = md.copy()
    return meta


def _write_pq_metadata_file(md_list, fs, path):
    """ Converts list of parquet metadata objects into
        a single shared _metadata file.
    """
    if md_list:
        metadata_path = fs.sep.join([path, "_metadata"])
        _meta = cudf.io.merge_parquet_filemetadata(md_list) if len(md_list) > 1 else md_list[0]
        with fs.open(metadata_path, "wb") as fil:
            _meta.tofile(fil)
    return


def _set_dtypes(chunk, dtypes):
    for col, dtype in dtypes.items():
        if type(dtype) is str:
            if "hex" in dtype and chunk[col].dtype == "object":
                chunk[col] = chunk[col].str.htoi()
                chunk[col] = chunk[col].astype(np.int32)
        else:
            chunk[col] = chunk[col].astype(dtype)
    return chunk


#
# Writer Definitions
#


def _writer_cls_factory(output_format, output_path):
    if output_format == "parquet":
        writer_cls = ParquetWriter
    elif output_format == "hugectr":
        writer_cls = HugeCTRWriter
    else:
        raise ValueError("Output format not yet supported.")

    fs = get_fs_token_paths(output_path)[0]
    return writer_cls, fs


def writer_factory(
    output_format,
    output_path,
    out_files_per_proc,
    shuffle,
    use_guid=False,
    bytes_io=False,
    num_threads=0,
):
    if output_format is None:
        return None

    writer_cls, fs = _writer_cls_factory(output_format, output_path)
    return writer_cls(
        output_path,
        num_out_files=out_files_per_proc,
        shuffle=shuffle,
        fs=fs,
        use_guid=use_guid,
        bytes_io=bytes_io,
        num_threads=num_threads,
    )


class Writer:
    def __init__(self):
        pass

    def add_data(self, gdf):
        raise NotImplementedError()

    def package_general_metadata(self):
        raise NotImplementedError()

    @classmethod
    def write_general_metadata(cls, data, fs, out_dir):
        raise NotImplementedError()

    @classmethod
    def write_special_metadata(cls, data, fs, out_dir):
        raise NotImplementedError()

    def close(self):
        pass


class ThreadedWriter(Writer):
    def __init__(
        self,
        out_dir,
        num_out_files=30,
        num_threads=0,
        cats=None,
        conts=None,
        labels=None,
        shuffle=None,
        fs=None,
        use_guid=False,
        bytes_io=False,
    ):
        # set variables
        self.out_dir = out_dir
        self.cats = cats
        self.conts = conts
        self.labels = labels
        self.shuffle = shuffle
        self.column_names = None
        if labels and conts:
            self.column_names = labels + conts

        self.col_idx = {}

        self.num_threads = num_threads
        self.num_out_files = num_out_files
        self.num_samples = [0] * num_out_files

        self.data_paths = None
        self.need_cal_col_names = True
        self.use_guid = use_guid
        self.bytes_io = bytes_io

        # Resolve file system
        self.fs = fs or get_fs_token_paths(str(out_dir))[0]

        # Only use threading if num_threads > 0
        if self.num_threads:

            # create thread queue and locks
            self.queue = queue.Queue(num_threads)
            self.write_locks = [threading.Lock() for _ in range(num_out_files)]

            # signifies that end-of-data and that the thread should shut down
            self._eod = object()

            # create and start threads
            for _ in range(num_threads):
                write_thread = threading.Thread(target=self._write_thread, daemon=True)
                write_thread.start()

    def set_col_names(self, labels, cats, conts):
        self.cats = cats
        self.conts = conts
        self.labels = labels
        self.column_names = labels + conts

    def _write_table(self, idx, data):
        return

    def _write_thread(self):
        return

    @annotate("add_data", color="orange", domain="nvt_python")
    def add_data(self, gdf):

        # Populate columns idxs
        if not self.col_idx:
            for i, x in enumerate(gdf.columns.values):
                self.col_idx[str(x)] = i

        nrows = gdf.shape[0]
        typ = np.min_scalar_type(nrows * 2)
        if self.shuffle and self.shuffle != "full":
            ind = cp.random.choice(np.arange(self.num_out_files, dtype=typ), nrows)
        else:
            ind = cp.arange(nrows, dtype=typ)
            np.floor_divide(ind, (nrows // self.num_out_files), out=ind)

        for x, df in group_split_dispatch(gdf, ind, self.num_out_files, ignore_index=True).items():
            self.num_samples[x] += len(df)
            if self.num_threads:
                self.queue.put((x, df))
            else:
                self._write_table(x, df)
                del df

        # wait for all writes to finish before exitting (so that we aren't using memory)
        if self.num_threads:
            self.queue.join()

    def package_general_metadata(self):
        data = {}
        if self.cats is None:
            return data
        data["data_paths"] = self.data_paths
        data["file_stats"] = []
        for i, path in enumerate(self.data_paths):
            fn = path.split(self.fs.sep)[-1]
            data["file_stats"].append({"file_name": fn, "num_rows": self.num_samples[i]})
        # cats
        data["cats"] = []
        for c in self.cats:
            data["cats"].append({"col_name": c, "index": self.col_idx[c]})
        # conts
        data["conts"] = []
        for c in self.conts:
            data["conts"].append({"col_name": c, "index": self.col_idx[c]})
        # labels
        data["labels"] = []
        for c in self.labels:
            data["labels"].append({"col_name": c, "index": self.col_idx[c]})

        return data

    @classmethod
    def write_general_metadata(cls, data, fs, out_dir):
        if not data:
            return
        data_paths = data.pop("data_paths", [])
        num_out_files = len(data_paths)

        # Write file_list
        file_list_writer = fs.open(fs.sep.join([out_dir, "_file_list.txt"]), "w")
        file_list_writer.write(str(num_out_files) + "\n")
        for f in data_paths:
            file_list_writer.write(f + "\n")
        file_list_writer.close()

        # Write metadata json
        metadata_writer = fs.open(fs.sep.join([out_dir, "_metadata.json"]), "w")
        json.dump(data, metadata_writer)
        metadata_writer.close()

    @classmethod
    def write_special_metadata(cls, data, fs, out_dir):
        pass

    def _close_writers(self):
        for writer in self.data_writers:
            writer.close()
        return None

    def close(self):
        if self.num_threads:
            # wake up all the worker threads and signal for them to exit
            for _ in range(self.num_threads):
                self.queue.put(self._eod)

            # wait for pending writes to finish
            self.queue.join()

        # Close writers and collect various metadata
        _general_meta = self.package_general_metadata()
        _special_meta = self._close_writers()

        # Move in-meomory file to disk
        if self.bytes_io:
            self._bytesio_to_disk()

        return _general_meta, _special_meta

    def _bytesio_to_disk(self):
        raise NotImplementedError("In-memory buffering/shuffling not implemented for this format.")


class ParquetWriter(ThreadedWriter):
    def __init__(self, out_dir, **kwargs):
        super().__init__(out_dir, **kwargs)
        self.data_paths = []
        self.data_writers = []
        self.data_bios = []
        for i in range(self.num_out_files):
            if self.use_guid:
                fn = f"{i}.{guid()}.parquet"
            else:
                fn = f"{i}.parquet"

            path = os.path.join(out_dir, fn)
            self.data_paths.append(path)
            if self.bytes_io:
                bio = BytesIO()
                self.data_bios.append(bio)
                self.data_writers.append(pwriter(bio, compression=None))
            else:
                self.data_writers.append(pwriter(path, compression=None))

    def _write_table(self, idx, data):
        self.data_writers[idx].write_table(data)

    def _write_thread(self):
        while True:
            item = self.queue.get()
            try:
                if item is self._eod:
                    break
                idx, data = item
                with self.write_locks[idx]:
                    self._write_table(idx, data)
            finally:
                self.queue.task_done()

    @classmethod
    def write_special_metadata(cls, md, fs, out_dir):
        # Sort metadata by file name and convert list of
        # tuples to a list of metadata byte-blobs
        md_list = [m[1] for m in sorted(list(md.items()), key=lambda x: natural_sort_key(x[0]))]

        # Aggregate metadata and write _metadata file
        _write_pq_metadata_file(md_list, fs, out_dir)

    def _close_writers(self):
        md_dict = {}
        for writer, path in zip(self.data_writers, self.data_paths):
            fn = path.split(self.fs.sep)[-1]
            md_dict[fn] = writer.close(metadata_file_path=fn)
        return md_dict

    def _bytesio_to_disk(self):
        for bio, path in zip(self.data_bios, self.data_paths):
            gdf = cudf.io.read_parquet(bio, index=False)
            bio.close()
            if self.shuffle == "full":
                gdf = _shuffle_gdf(gdf)
            gdf.to_parquet(path, compression=None, index=False)
        return


class HugeCTRWriter(ThreadedWriter):
    def __init__(self, out_dir, **kwargs):
        super().__init__(out_dir, **kwargs)
        self.data_paths = [os.path.join(out_dir, f"{i}.data") for i in range(self.num_out_files)]
        self.data_writers = [open(f, "ab") for f in self.data_paths]

    def _write_table(self, idx, data):
        ones = np.array(([1] * data.shape[0]), dtype=np.intc)
        df = data[self.column_names].to_pandas().astype(np.single)
        for i in range(len(self.cats)):
            df["___" + str(i) + "___" + self.cats[i]] = ones
            df[self.cats[i]] = data[self.cats[i]].to_pandas().astype(np.longlong)
            self.data_writers[idx].write(df.to_numpy().tobytes())

    def _write_thread(self):
        while True:
            item = self.queue.get()
            try:
                if item is self._eod:
                    break
                idx, data = item
                with self.write_locks[idx]:
                    self._write_table(idx, data)
            finally:
                self.queue.task_done()

    def _close_writers(self):
        for i, writer in enumerate(self.data_writers):
            if self.cats:
                # Write HugeCTR Metadata
                writer.seek(0)
                # error_check (0: no error check; 1: check_num)
                # num of samples in this file
                # Dimension of the labels
                # Dimension of the features
                # slot_num for each embedding
                # reserved for future use
                header = np.array(
                    [
                        0,
                        self.num_samples[i],
                        len(self.labels),
                        len(self.conts),
                        len(self.cats),
                        0,
                        0,
                        0,
                    ],
                    dtype=np.longlong,
                )
                writer.write(header.tobytes())
            writer.close()
        return None


#
# Dask-based IO
#


@annotate("write_output_partition", color="green", domain="nvt_python")
def _write_output_partition(
    gdf,
    processed_path,
    shuffle,
    out_files_per_proc,
    fs,
    cat_names,
    cont_names,
    label_names,
    output_format,
    num_threads,
):
    gdf_size = len(gdf)
    out_files_per_proc = out_files_per_proc or 1

    # Get cached writer (or create/cache a new one)
    with get_worker_cache("writer") as writer_cache:
        writer = writer_cache.get(processed_path, None)
        if writer is None:
            writer = writer_factory(
                output_format,
                processed_path,
                out_files_per_proc,
                shuffle,
                use_guid=True,
                bytes_io=(shuffle == "full"),
                num_threads=num_threads,
            )
            writer.set_col_names(labels=label_names, cats=cat_names, conts=cont_names)
            writer_cache[processed_path] = writer

        # Add data
        writer.add_data(gdf)

    return gdf_size


def _ddf_to_dataset(
    ddf,
    fs,
    output_path,
    shuffle,
    out_files_per_proc,
    cat_names,
    cont_names,
    label_names,
    output_format,
    client,
    num_threads,
):
    # Construct graph for Dask-based dataset write
    name = "write-processed"
    write_name = name + tokenize(
        ddf, shuffle, out_files_per_proc, cat_names, cont_names, label_names
    )
    task_list = []
    dsk = {}
    for idx in range(ddf.npartitions):
        key = (write_name, idx)
        dsk[key] = (
            _write_output_partition,
            (ddf._name, idx),
            output_path,
            shuffle,
            out_files_per_proc,
            fs,
            cat_names,
            cont_names,
            label_names,
            output_format,
            num_threads,
        )
        task_list.append(key)
    dsk[name] = (lambda x: x, task_list)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[ddf])
    out = Delayed(name, graph)

    # Trigger write execution
    if client:
        out = client.compute(out).result()
    else:
        out = dask.compute(out, scheduler="threads", num_workers=1)[0]

    # Follow-up Shuffling and _metadata creation
    _finish_dataset(client, ddf, output_path, fs, output_format)


def _finish_dataset(client, ddf, output_path, fs, output_format):
    # Finish data writing
    if client:
        client.cancel(ddf)
        ddf = None
        out = client.run(_worker_finish, output_path)

        general_md = []
        special_md = []
        for (gen, spec) in out.values():
            general_md.append(gen)
            special_md.append(spec)

        general_md = _merge_general_metadata(general_md)
        special_md = dict(collections.ChainMap(*special_md))
    else:
        ddf = None
        general_md, special_md = _worker_finish(output_path)

    # Write metadata on client
    wc, fs = _writer_cls_factory(output_format, output_path)
    wc.write_general_metadata(general_md, fs, output_path)
    wc.write_special_metadata(special_md, fs, output_path)

    # Clean writer caches
    if client:
        client.run(clean_worker_cache, "writer")
    else:
        clean_worker_cache("writer")


def _worker_finish(processed_path):
    general_md, special_md = {}, {}
    with get_worker_cache("writer") as writer_cache:
        writer = writer_cache.get(processed_path, None)
        if writer:
            general_md, special_md = writer.close()

    return general_md, special_md


class Dataset:
    """ Dask-based Dataset Class
        Converts a dataset into a dask_cudf DataFrame on demand

    Parameters
    -----------
    path_or_source : str, list of str, or <dask.dataframe|cudf|pd>.DataFrame
        Dataset path (or list of paths), or a DataFrame. If string,
        should specify a specific file or directory path. If this is a
        directory path, the directory structure must be flat (nested
        directories are not yet supported).
    engine : str or DatasetEngine
        DatasetEngine object or string identifier of engine. Current
        string options include: ("parquet", "csv"). This argument
        is ignored if path_or_source is a DataFrame type.
    part_size : str or int
        Desired size (in bytes) of each Dask partition.
        If None, part_mem_fraction will be used to calculate the
        partition size.  Note that the underlying engine may allow
        other custom kwargs to override this argument. This argument
        is ignored if path_or_source is a DataFrame type.
    part_mem_fraction : float (default 0.125)
        Fractional size of desired dask partitions (relative
        to GPU memory capacity). Ignored if part_size is passed
        directly. Note that the underlying engine may allow other
        custom kwargs to override this argument. This argument
        is ignored if path_or_source is a DataFrame type.
    storage_options: None or dict
        Further parameters to pass to the bytes backend. This argument
        is ignored if path_or_source is a DataFrame type.
    """

    def __init__(
        self,
        path_or_source,
        engine=None,
        part_size=None,
        part_mem_fraction=None,
        storage_options=None,
        dtypes=None,
        **kwargs,
    ):
        self.dtypes = dtypes
        if isinstance(path_or_source, (dask.dataframe.DataFrame, cudf.DataFrame, pd.DataFrame)):
            # User is passing in a <dask.dataframe|cudf|pd>.DataFrame
            # Use DataFrameDatasetEngine
            if isinstance(path_or_source, cudf.DataFrame):
                path_or_source = dask_cudf.from_cudf(path_or_source, npartitions=1)
            elif isinstance(path_or_source, pd.DataFrame):
                path_or_source = dask_cudf.from_cudf(
                    cudf.from_pandas(path_or_source), npartitions=1
                )
            elif not isinstance(path_or_source, dask_cudf.DataFrame):
                path_or_source = dask_cudf.from_dask_dataframe(path_or_source)
            if part_size:
                warnings.warn("part_size is ignored for DataFrame input.")
            if part_mem_fraction:
                warnings.warn("part_mem_fraction is ignored for DataFrame input.")
            self.engine = DataFrameDatasetEngine(path_or_source)
        else:
            if part_size:
                # If a specific partition size is given, use it directly
                part_size = parse_bytes(part_size)
            else:
                # If a fractional partition size is given, calculate part_size
                part_mem_fraction = part_mem_fraction or 0.125
                assert part_mem_fraction > 0.0 and part_mem_fraction < 1.0
                if part_mem_fraction > 0.25:
                    warnings.warn(
                        "Using very large partitions sizes for Dask. "
                        "Memory-related errors are likely."
                    )
                part_size = int(device_mem_size(kind="total") * part_mem_fraction)

            # Engine-agnostic path handling
            paths = path_or_source
            if hasattr(paths, "name"):
                paths = stringify_path(paths)
            if isinstance(paths, str):
                paths = [paths]

            storage_options = storage_options or {}
            # If engine is not provided, try to infer from end of paths[0]
            if engine is None:
                engine = paths[0].split(".")[-1]
            if isinstance(engine, str):
                if engine == "parquet":
                    self.engine = ParquetDatasetEngine(
                        paths, part_size, storage_options=storage_options, **kwargs
                    )
                elif engine == "csv":
                    self.engine = CSVDatasetEngine(
                        paths, part_size, storage_options=storage_options, **kwargs
                    )
                else:
                    raise ValueError("Only parquet and csv supported (for now).")
            else:
                self.engine = engine(paths, part_size, storage_options=storage_options)

    def to_ddf(self, columns=None):
        ddf = self.engine.to_ddf(columns=columns)
        if self.dtypes:
            _meta = _set_dtypes(ddf._meta, self.dtypes)
            return ddf.map_partitions(_set_dtypes, self.dtypes, meta=_meta)
        return ddf

    def to_iter(self, columns=None):
        if isinstance(columns, str):
            columns = [columns]

        return DataFrameIter(self.to_ddf(columns=columns))

    @property
    def num_rows(self):
        return self.engine.num_rows


class DatasetEngine:
    """ DatasetEngine Class

        Base class for Dask-powered IO engines. Engines must provide
        a ``to_ddf`` method.
    """

    def __init__(self, paths, part_size, storage_options=None):
        paths = sorted(paths, key=natural_sort_key)
        self.paths = paths
        self.part_size = part_size

        fs, fs_token, _ = get_fs_token_paths(paths, mode="rb", storage_options=storage_options)
        self.fs = fs
        self.fs_token = fs_token

    def to_ddf(self, columns=None):
        raise NotImplementedError(""" Return a dask_cudf.DataFrame """)

    @property
    def num_rows(self):
        raise NotImplementedError(""" Returns the number of rows in the dataset """)


class ParquetDatasetEngine(DatasetEngine):
    """ ParquetDatasetEngine

        Dask-based version of cudf.read_parquet.
    """

    def __init__(
        self,
        paths,
        part_size,
        storage_options,
        row_groups_per_part=None,
        legacy=False,
        batch_size=None,
    ):
        # TODO: Improve dask_cudf.read_parquet performance so that
        # this class can be slimmed down.
        super().__init__(paths, part_size, storage_options)
        self.batch_size = batch_size
        self._metadata, self._base = self.metadata
        self._pieces = None
        if row_groups_per_part is None:
            file_path = self._metadata.row_group(0).column(0).file_path
            path0 = (
                self.fs.sep.join([self._base, file_path])
                if file_path != ""
                else self._base  # This is a single file
            )

        if row_groups_per_part is None:
            rg_byte_size_0 = (
                cudf.io.read_parquet(path0, row_groups=0, row_group=0)
                .memory_usage(deep=True, index=True)
                .sum()
            )
            row_groups_per_part = self.part_size / rg_byte_size_0
            if row_groups_per_part < 1.0:
                warnings.warn(
                    f"Row group size {rg_byte_size_0} is bigger than requested part_size "
                    f"{self.part_size}"
                )
                row_groups_per_part = 1.0

        self.row_groups_per_part = int(row_groups_per_part)

        assert self.row_groups_per_part > 0

    @property
    def pieces(self):
        if self._pieces is None:
            self._pieces = self._get_pieces(self._metadata, self._base)
        return self._pieces

    @property
    @functools.lru_cache(1)
    def metadata(self):
        paths = self.paths
        fs = self.fs
        if len(paths) > 1:
            # This is a list of files
            dataset = pq.ParquetDataset(paths, filesystem=fs, validate_schema=False)
            base, fns = _analyze_paths(paths, fs)
        elif fs.isdir(paths[0]):
            # This is a directory
            dataset = pq.ParquetDataset(paths[0], filesystem=fs, validate_schema=False)
            allpaths = fs.glob(paths[0] + fs.sep + "*")
            base, fns = _analyze_paths(allpaths, fs)
        else:
            # This is a single file
            dataset = pq.ParquetDataset(paths[0], filesystem=fs)
            base = paths[0]
            fns = [None]

        metadata = None
        if dataset.metadata:
            # We have a metadata file
            return dataset.metadata, base
        else:
            # Collect proper metadata manually
            metadata = None
            for piece, fn in zip(dataset.pieces, fns):
                md = piece.get_metadata()
                if fn:
                    md.set_file_path(fn)
                if metadata:
                    metadata.append_row_groups(md)
                else:
                    metadata = md
            return metadata, base

    @property
    def num_rows(self):
        metadata, _ = self.metadata
        return metadata.num_rows

    @annotate("get_pieces", color="green", domain="nvt_python")
    def _get_pieces(self, metadata, data_path):

        # get the number of row groups per file
        file_row_groups = defaultdict(int)
        for rg in range(metadata.num_row_groups):
            fpath = metadata.row_group(rg).column(0).file_path
            if fpath is None:
                raise ValueError("metadata is missing file_path string.")
            file_row_groups[fpath] += 1

        # create pieces from each file, limiting the number of row_groups in each piece
        pieces = []
        for filename, row_group_count in file_row_groups.items():
            row_groups = range(row_group_count)
            for i in range(0, row_group_count, self.row_groups_per_part):
                rg_list = list(row_groups[i : i + self.row_groups_per_part])
                full_path = (
                    self.fs.sep.join([data_path, filename])
                    if filename != ""
                    else data_path  # This is a single file
                )
                pieces.append((full_path, rg_list))
        return pieces

    @staticmethod
    @annotate("read_piece", color="green", domain="nvt_python")
    def read_piece(piece, columns):
        path, row_groups = piece
        return cudf.io.read_parquet(path, row_groups=row_groups, columns=columns, index=False)

    def meta_empty(self, columns=None):
        path, _ = self.pieces[0]
        return cudf.io.read_parquet(path, row_groups=0, columns=columns, index=False).iloc[:0]

    def to_ddf(self, columns=None):
        pieces = self.pieces
        name = "parquet-to-ddf-" + tokenize(self.fs_token, pieces, columns)
        dsk = {
            (name, p): (ParquetDatasetEngine.read_piece, piece, columns)
            for p, piece in enumerate(pieces)
        }
        meta = self.meta_empty(columns=columns)
        divisions = [None] * (len(pieces) + 1)
        return new_dd_object(dsk, name, meta, divisions)


class CSVDatasetEngine(DatasetEngine):
    """ CSVDatasetEngine

        Thin wrapper around dask_cudf.read_csv.
    """

    def __init__(self, paths, part_size, storage_options=None, **kwargs):
        super().__init__(paths, part_size, storage_options)
        self._meta = {}
        self.csv_kwargs = kwargs
        self.csv_kwargs["storage_options"] = storage_options

        # CSV reader needs a list of files
        # (Assume flat directory structure if this is a dir)
        if len(self.paths) == 1 and self.fs.isdir(self.paths[0]):
            self.paths = self.fs.glob(self.fs.sep.join([self.paths[0], "*"]))

    def to_ddf(self, columns=None):
        if columns:
            return dask_cudf.read_csv(self.paths, chunksize=self.part_size, **self.csv_kwargs)[
                columns
            ]
        return dask_cudf.read_csv(self.paths, chunksize=self.part_size, **self.csv_kwargs)


class DataFrameDatasetEngine(DatasetEngine):
    """ DataFrameDatasetEngine

        Allow NVT to interact with a dask_cudf.DataFrame object
        in the same way as a dataset on disk.
    """

    def __init__(self, ddf):
        self._ddf = ddf

    def to_ddf(self, columns=None):
        if isinstance(columns, list):
            return self._ddf[columns]
        elif isinstance(columns, str):
            return self._ddf[[columns]]
        return self._ddf

    @property
    def num_rows(self):
        return len(self._ddf)


class DataFrameIter:
    def __init__(self, ddf, columns=None):
        self._ddf = ddf
        self.columns = columns

    def __len__(self):
        return self._ddf.npartitions

    def __iter__(self):
        for part in self._ddf.partitions:
            if self.columns:
                yield part[self.columns].compute(scheduler="threads", num_workers=1)
            else:
                yield part.compute(scheduler="threads", num_workers=1)
            part = None
