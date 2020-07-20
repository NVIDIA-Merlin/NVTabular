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
import contextlib
import functools
import io
import json
import logging
import math
import os
import queue
import threading
import warnings
from collections import defaultdict
from io import BytesIO
from itertools import islice
from uuid import uuid4

import cudf
import cupy as cp
import dask_cudf
import numba.cuda as cuda
import numpy as np
import pyarrow.parquet as pq
from cudf._lib.nvtx import annotate
from cudf.io.parquet import ParquetWriter as pwriter
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.dataframe.io.parquet.utils import _analyze_paths
from dask.delayed import Delayed
from dask.distributed import get_worker
from dask.highlevelgraph import HighLevelGraph
from dask.utils import natural_sort_key, parse_bytes
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path

# Use global variable as the default
# cache when there are no distributed workers
_DEFAULT_CACHE = None
_DEFAULT_CACHE_LOCK = threading.RLock()


LOG = logging.getLogger("nvtabular")


#
# Cache-Specific Code
#


class WriterCache:
    def __init__(self):
        self.pq_writer_cache = {}

    def __del__(self):
        for path, (pw, fpath) in self.pq_writer_cache.items():
            pw.close()

    def get_pq_writer(self, prefix, s, mem):
        pw, fil = self.pq_writer_cache.get(prefix, (None, None))
        if pw is None:
            if mem:
                fil = BytesIO()
                pw = pwriter(fil, compression=None)
                self.pq_writer_cache[prefix] = (pw, fil)
            else:
                outfile_id = guid() + ".parquet"
                full_path = ".".join([prefix, outfile_id])
                pw = pwriter(full_path, compression=None)
                self.pq_writer_cache[prefix] = (pw, full_path)
        return pw


@contextlib.contextmanager
def get_cache():
    with _DEFAULT_CACHE_LOCK:
        yield _get_cache()


def _get_cache():
    try:
        worker = get_worker()
    except ValueError:
        # There is no dask.distributed worker.
        # Assume client/worker are same process
        global _DEFAULT_CACHE
        if _DEFAULT_CACHE is None:
            _DEFAULT_CACHE = WriterCache()
        return _DEFAULT_CACHE
    if not hasattr(worker, "pw_cache"):
        worker.pw_cache = WriterCache()
    return worker.pw_cache


def clean_pw_cache():
    with _DEFAULT_CACHE_LOCK:
        try:
            worker = get_worker()
        except ValueError:
            global _DEFAULT_CACHE
            if _DEFAULT_CACHE is not None:
                del _DEFAULT_CACHE
                _DEFAULT_CACHE = None
            return
        if hasattr(worker, "pw_cache"):
            del worker.pw_cache
        return


def close_cached_pw(fs):
    md_dict = {}
    with get_cache() as cache:
        for path, (pw, bio) in cache.pq_writer_cache.items():
            fn = bio.split(fs.sep)[-1]
            md_dict[fn] = pw.close(metadata_file_path=fn)
    return md_dict


#
# Helper Function definitions
#


def _allowable_batch_size(gpu_memory_frac, row_size):
    free_mem = device_mem_size(kind="free")
    gpu_memory = free_mem * gpu_memory_frac
    return max(int(gpu_memory / row_size), 1)


def _get_read_engine(engine, file_path, **kwargs):
    fs, _, _ = get_fs_token_paths(file_path, mode="rb")
    LOG.debug("opening '%s' as %s", file_path, engine)
    if engine is None:
        engine = file_path.split(".")[-1]
    if not isinstance(engine, str):
        raise TypeError("Expecting engine as string type.")

    if engine == "csv":
        return CSVFileReader(file_path, fs, **kwargs)
    elif engine == "parquet":
        return PQFileReader(file_path, fs, **kwargs)
    else:
        raise ValueError("Unrecognized read engine.")


def _shuffle_gdf(gdf, gdf_size=None):
    """ Shuffles a cudf dataframe, returning a new dataframe with randomly
    ordered rows """
    gdf_size = gdf_size or len(gdf)
    arr = cp.arange(gdf_size)
    cp.random.shuffle(arr)
    return gdf.iloc[arr]


#
# GPUFileReader Base Class
#


class GPUFileReader:
    def __init__(
        self, file_path, fs, gpu_memory_frac, batch_size=None, row_size=None, columns=None, **kwargs
    ):
        """ GPUFileReader Constructor
        """
        self.file_path = file_path
        self.fs = fs
        self.row_size = row_size
        self.columns = columns
        self.intialize_reader(gpu_memory_frac, batch_size, **kwargs)

    def intialize_reader(self, **kwargs):
        """ Define necessary file statistics and properties for reader
        """
        raise NotImplementedError()

    def __iter__(self):
        """ Iterates through the file, yielding a series of cudf.DataFrame objects
        """
        raise NotImplementedError()

    def __len__(self):
        """ Returns the number of dataframe chunks in the file """
        raise NotImplementedError()

    @property
    def estimated_row_size(self):
        return self.row_size


#
# GPUFileReader Sub Classes (Parquet and CSV Engines)
#


class PQFileReader(GPUFileReader):
    def intialize_reader(self, gpu_memory_frac, batch_size, **kwargs):
        self.reader = cudf.read_parquet

        # Read Parquet-file metadata
        (self.num_rows, self.num_row_groups, columns) = cudf.io.read_parquet_metadata(
            self.fs.open(self.file_path)
        )
        # Use first row-group metadata to estimate memory-rqs
        # NOTE: We could also use parquet metadata here, but
        #       `total_uncompressed_size` for each column is
        #       not representative of dataframe size for
        #       strings/categoricals (parquet only stores uniques)
        self.row_size = self.row_size or 0
        if self.num_rows > 0 and self.row_size == 0:
            for col in self.reader(self.file_path, num_rows=1)._columns:
                # removed logic for max in first x rows, it was
                # causing infinite loops for our customers on their datasets.
                self.row_size += col.dtype.itemsize
        # Check if we are using row groups
        self.use_row_groups = kwargs.get("use_row_groups", None)
        self.row_group_batch = 1
        self.next_row_group = 0

        # Determine batch size if needed
        if batch_size and not self.use_row_groups:
            self.batch_size = batch_size
            self.use_row_groups = False
        else:
            # Use row size to calculate "allowable" batch size
            gpu_memory_batch = _allowable_batch_size(gpu_memory_frac, self.row_size)
            self.batch_size = min(gpu_memory_batch, self.num_rows)

            # Use row-groups if they meet memory constraints
            rg_size = int(self.num_rows / self.num_row_groups)
            if (self.use_row_groups is None) and (rg_size <= gpu_memory_batch):
                self.use_row_groups = True
            elif self.use_row_groups is None:
                self.use_row_groups = False

            # Determine row-groups per batch
            if self.use_row_groups:
                self.row_group_batch = max(int(gpu_memory_batch / rg_size), 1)

    def __len__(self):
        return int((self.num_rows + self.batch_size - 1) // self.batch_size)

    def __iter__(self):
        for nskip in range(0, self.num_rows, self.batch_size):
            # not using row groups because concat uses up double memory
            # making iterator unable to use selected gpu memory fraction.
            batch = min(self.batch_size, self.num_rows - nskip)
            LOG.debug(
                "loading chunk from %s, (skip_rows=%s, num_rows=%s)", self.file_path, nskip, batch
            )
            gdf = self.reader(
                self.file_path, num_rows=batch, skip_rows=nskip, engine="cudf", columns=self.columns
            )
            gdf.reset_index(drop=True, inplace=True)
            yield gdf
            gdf = None


class CSVFileReader(GPUFileReader):
    def intialize_reader(self, gpu_memory_frac, batch_size, **kwargs):
        self.reader = cudf.read_csv
        # Count rows and determine column names
        estimate_row_size = False
        if self.row_size is None:
            self.row_size = 0
            estimate_row_size = True
        self.offset = 0
        self.file_bytes = self.fs.stat(str(self.file_path))["size"]

        # Use first row to estimate memory-reqs
        names = kwargs.get("names", None)
        dtype = kwargs.get("dtype", None)
        # default csv delim is ","
        sep = kwargs.get("sep", ",")
        self.sep = sep
        self.names = []
        dtype_inf = {}
        nrows = 10
        head = "".join(islice(self.fs.open(self.file_path, "r"), nrows))
        snippet = self.reader(
            io.StringIO(head), nrows=nrows, names=names, dtype=dtype, sep=sep, header=0
        )
        self.inferred_names = not names
        if self.file_bytes > 0:
            for i, col in enumerate(snippet.columns):
                if names:
                    name = names[i]
                else:
                    name = col
                self.names.append(name)
            for i, col in enumerate(snippet._columns):
                if estimate_row_size:
                    self.row_size += col.dtype.itemsize
                dtype_inf[self.names[i]] = col.dtype
        self.dtype = dtype or dtype_inf

        # Determine batch size if needed
        if batch_size:
            self.batch_size = batch_size * self.row_size
        else:
            free_mem = device_mem_size(kind="free")
            self.batch_size = free_mem * gpu_memory_frac
        self.num_chunks = int((self.file_bytes + self.batch_size - 1) // self.batch_size)

    def __len__(self):
        return self.num_chunks

    def __iter__(self):
        for chunks in range(self.num_chunks):
            LOG.debug(
                "loading chunk from %s, byte_range=%s",
                self.file_path,
                (chunks * self.batch_size, self.batch_size),
            )
            chunk = self.reader(
                self.file_path,
                byte_range=(chunks * self.batch_size, self.batch_size),
                names=self.names,
                header=0 if chunks == 0 and self.inferred_names else None,
                sep=self.sep,
            )

            if self.columns:
                for col in self.columns:
                    chunk[col] = chunk[col].astype(self.dtype[col])
                chunk = chunk[self.columns]

            yield chunk
            chunk = None


#
# GPUFileIterator (Single File Iterator)
#


class GPUFileIterator:
    def __init__(
        self,
        file_path,
        engine=None,
        gpu_memory_frac=0.5,
        batch_size=None,
        columns=None,
        use_row_groups=None,
        dtypes=None,
        names=None,
        row_size=None,
        fs=None,
        **kwargs,
    ):
        self.file_path = file_path
        self.engine = _get_read_engine(
            engine,
            file_path,
            columns=columns,
            batch_size=batch_size,
            gpu_memory_frac=gpu_memory_frac,
            use_row_groups=use_row_groups,
            dtypes=dtypes,
            names=names,
            row_size=None,
            **kwargs,
        )
        self.dtypes = dtypes
        self.columns = columns

    def __iter__(self):
        for chunk in self.engine:
            if self.dtypes:
                self._set_dtypes(chunk)
            yield chunk
            chunk = None

    def __len__(self):
        return len(self.engine)

    def _set_dtypes(self, chunk):
        for col, dtype in self.dtypes.items():
            if type(dtype) is str:
                if "hex" in dtype and chunk[col].dtype == "object":
                    chunk[col] = chunk[col].str.htoi()
                    chunk[col] = chunk[col].astype(np.int32)
            else:
                chunk[col] = chunk[col].astype(dtype)


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
        num_threads=4,
        cats=None,
        conts=None,
        labels=None,
        shuffle=None,
        fs=None,
    ):
        # set variables
        self.hugectr_bin = False
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

        # Resolve file system
        self.fs = fs or get_fs_token_paths(str(out_dir))[0]

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

    def _write_thread(self):
        return

    @annotate("add_data", color="orange", domain="nvt_python")
    def add_data(self, gdf):

        # Shuffle if necessary
        if self.shuffle:
            gdf = _shuffle_gdf(gdf)

        # Populate columns idxs
        if not self.col_idx:
            for i, x in enumerate(gdf.columns.values):
                self.col_idx[str(x)] = i

        # get slice info
        int_slice_size = gdf.shape[0] // self.num_out_files
        slice_size = int_slice_size if gdf.shape[0] % int_slice_size == 0 else int_slice_size + 1

        for x in range(self.num_out_files):
            start = x * slice_size
            end = start + slice_size
            # check if end is over length
            end = end if end <= gdf.shape[0] else gdf.shape[0]
            to_write = gdf.iloc[start:end]
            self.num_samples[x] = self.num_samples[x] + to_write.shape[0]
            self.queue.put((x, to_write))

        # wait for all writes to finish before exitting (so that we aren't using memory)
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
        file_list_writer = fs.open(fs.sep.join([out_dir, "file_list.txt"]), "w")
        file_list_writer.write(str(num_out_files) + "\n")
        for f in data_paths:
            file_list_writer.write(f + "\n")
        file_list_writer.close()

        # Write metadata json
        metadata_writer = fs.open(fs.sep.join([out_dir, "metadata.json"]), "w")
        json.dump(data, metadata_writer)
        metadata_writer.close()

    @classmethod
    def write_special_metadata(cls, data, fs, out_dir):
        pass

    def _close_writers(self):
        for writer in self.data_writers:
            writer.close()
        return None

    def close(self, write_metadata=True):
        # wake up all the worker threads and signal for them to exit
        for _ in range(self.num_threads):
            self.queue.put(self._eod)

        # wait for pending writes to finish
        self.queue.join()

        # Close writers and collect various metadata
        _gen_meta = self.package_general_metadata()
        _special_meta = self._close_writers()
        return _gen_meta, _special_meta


class ParquetWriter(ThreadedWriter):
    def __init__(self, out_dir, use_guid=False, **kwargs):
        super().__init__(out_dir, **kwargs)
        self.data_fns = []
        self.data_paths = []
        self.data_writers = []
        for i in range(self.num_out_files):
            if use_guid:
                fn = f"{i}.{guid()}.parquet"
            else:
                fn = f"{i}.parquet"
            path = os.path.join(out_dir, fn)
            self.data_fns.append(fn)
            self.data_paths.append(path)
            self.data_writers.append(pwriter(path, compression=None))

    def _write_thread(self):
        while True:
            item = self.queue.get()
            try:
                if item is self._eod:
                    break
                idx, data = item
                with self.write_locks[idx]:
                    self.data_writers[idx].write_table(data)
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
        for writer, fn in zip(self.data_writers, self.data_fns):
            md_dict[fn] = writer.close(metadata_file_path=fn)
        return md_dict


class HugeCTRWriter(ThreadedWriter):
    def __init__(self, out_dir, **kwargs):
        super().__init__(out_dir, **kwargs)
        self.data_paths = [os.path.join(out_dir, f"{i}.data") for i in range(self.num_out_files)]
        self.data_writers = [open(f, "ab") for f in self.data_paths]
        self.hugectr_bin = True

    def _write_thread(self):
        while True:
            item = self.queue.get()
            try:
                if item is self._eod:
                    break
                idx, data = item
                with self.write_locks[idx]:
                    ones = np.array(([1] * data.shape[0]), dtype=np.intc)
                    df = data[self.column_names].to_pandas().astype(np.single)
                    for i in range(len(self.cats)):
                        df["___" + str(i) + "___" + self.cats[i]] = ones
                        df[self.cats[i]] = data[self.cats[i]].to_pandas().astype(np.longlong)
                        self.data_writers[idx].write(df.to_numpy().tobytes())
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


@annotate("write_output_partition", color="green", domain="nvt_python")
def _write_output_partition(gdf, processed_path, shuffle, out_files_per_proc, fs):
    gdf_size = len(gdf)
    out_files_per_proc = out_files_per_proc or 1
    if shuffle and shuffle != "full":
        # We should do a real sort here
        gdf = _shuffle_gdf(gdf, gdf_size=gdf_size)
    split_size = math.ceil(gdf_size / out_files_per_proc)
    ind = cp.ones(gdf_size).cumsum() // split_size
    del ind
    for s in range(out_files_per_proc):
        prefix = fs.sep.join([processed_path, "split." + str(s)])
        with get_cache() as cache:
            pw = cache.get_pq_writer(prefix, s, mem=(shuffle == "full"))
            pw.write_table(gdf.iloc[s * split_size : (s + 1) * split_size])

    return gdf_size


def _ddf_to_pq_dataset(ddf, fs, output_path, shuffle, out_files_per_proc):
    name = "write-processed"
    write_name = name + tokenize(ddf, shuffle, out_files_per_proc)
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
        )
        task_list.append(key)
    dsk[name] = (lambda x: x, task_list)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[ddf])
    return Delayed(name, graph)


def _finish_pq_dataset(client, ddf, shuffle, output_path, fs):
    # Deal with "full" (per-worker) shuffle here.
    if shuffle == "full":
        if client:
            client.cancel(ddf)
            ddf = None
            worker_md = client.run(_worker_shuffle, output_path, fs)
            worker_md = list(collections.ChainMap(*worker_md.values()).items())
        else:
            ddf = None
            worker_md = _worker_shuffle(output_path, fs)
            worker_md = list(worker_md.items())
    else:
        # Collect parquet metadata while closing
        # ParquetWriter object(s)
        if client:
            worker_md = client.run(close_cached_pw, fs)
            worker_md = list(collections.ChainMap(*worker_md.values()).items())
        else:
            worker_md = close_cached_pw(fs)
            worker_md = list(worker_md.items())

    # Sort metadata by file name and convert list of
    # tuples to a list of metadata byte-blobs
    md_list = [m[1] for m in sorted(worker_md, key=lambda x: natural_sort_key(x[0]))]

    # Aggregate metadata and write _metadata file
    _write_pq_metadata_file(md_list, fs, output_path)

    # Close ParquetWriter Objects
    if client:
        client.run(clean_pw_cache)
    else:
        clean_pw_cache()


def _write_pq_metadata_file(md_list, fs, path):
    if md_list:
        metadata_path = fs.sep.join([path, "_metadata"])
        _meta = cudf.io.merge_parquet_filemetadata(md_list) if len(md_list) > 1 else md_list[0]
        with fs.open(metadata_path, "wb") as fil:
            _meta.tofile(fil)
    return


@annotate("worker_shuffle", color="green", domain="nvt_python")
def _worker_shuffle(processed_path, fs):
    metadata_dict = {}
    with get_cache() as cache:
        for path, (pw, bio) in cache.pq_writer_cache.items():
            pw.close()

            gdf = cudf.io.read_parquet(bio, index=False)
            bio.close()

            gdf = _shuffle_gdf(gdf)
            rel_path = "shuffled.%s.parquet" % (guid())
            full_path = fs.sep.join([processed_path, rel_path])
            metadata_dict[rel_path] = gdf.to_parquet(
                full_path, compression=None, index=False, metadata_file_path=rel_path
            )
        return metadata_dict


class Dataset:
    """ Dask-based Dataset Class
        Converts a dataset into a dask_cudf DataFrame on demand

    Parameters
    -----------
    path : str or list of str
        Dataset path (or list of paths). If string, should specify
        a specific file or directory path. If this is a directory
        path, the directory structure must be flat (nested directories
        are not yet supported).
    engine : str or DatasetEngine
        DatasetEngine object or string identifier of engine. Current
        string options include: ("parquet").
    part_size : str or int
        Desired size (in bytes) of each Dask partition.
        If None, part_mem_fraction will be used to calculate the
        partition size.  Note that the underlying engine may allow
        other custom kwargs to override this argument.
    part_mem_fraction : float (default 0.125)
        Fractional size of desired dask partitions (relative
        to GPU memory capacity). Ignored if part_size is passed
        directly. Note that the underlying engine may allow other
        custom kwargs to override this argument.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.
    """

    def __init__(
        self,
        paths,
        engine=None,
        part_size=None,
        part_mem_fraction=None,
        storage_options=None,
        **kwargs,
    ):
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
        return self.engine.to_ddf(columns=columns)

    def to_iter(self, columns=None):
        return self.engine.to_iter(columns=columns)

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

    def to_iter(self, columns=None):
        raise NotImplementedError(""" Return a Iterator over the cudf chunks of the dataset  """)

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
        return cudf.io.read_parquet(
            path,
            # cudf 0.15 (only need `row_groups`)
            row_groups=row_groups,
            # cudf 0.14 (need `row_group` and `row_group_count`)
            row_group=row_groups[0],
            row_group_count=len(row_groups),
            columns=columns,
            index=False,
        )

    def meta_empty(self, columns=None):
        path, _ = self.pieces[0]
        return cudf.io.read_parquet(
            path,
            # cudf 0.15
            row_groups=0,
            # cudf 0.14
            row_group=0,
            columns=columns,
            index=False,
        ).iloc[:0]

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

    def to_iter(self, columns=None):
        part_mem_fraction = self.part_size / device_mem_size(kind="total")
        for path in self.paths:
            yield from PQFileReader(
                path, self.fs, part_mem_fraction, columns=columns, batch_size=self.batch_size
            )


class CSVDatasetEngine(DatasetEngine):
    """ CSVDatasetEngine

        Thin wrapper around dask_cudf.read_csv.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self._meta = {}
        self.csv_kwargs = kwargs
        # CSV reader needs a list of files
        # (Assume flat directory structure if this is a dir)
        if len(self.paths) == 1 and self.fs.isdir(self.paths[0]):
            self.paths = self.fs.glob(self.fs.sep.join([self.paths[0], "*"]))

    def to_ddf(self, columns=None):
        return dask_cudf.read_csv(self.paths, chunksize=self.part_size, **self.csv_kwargs)[columns]

    def to_iter(self, columns=None):
        part_mem_fraction = self.part_size / device_mem_size(kind="total")
        for path in self.paths:
            yield from CSVFileReader(
                path, self.fs, part_mem_fraction, columns=columns, **self.csv_kwargs
            )
