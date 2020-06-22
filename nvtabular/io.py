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

import io
import json
import logging
import os
import queue
import threading
import warnings
from itertools import islice

import cudf
import cupy as cp
import numba.cuda as cuda
import numpy as np
from cudf._lib.nvtx import annotate
from cudf.io.parquet import ParquetWriter as pwriter

LOG = logging.getLogger("nvtabular")

#
# Helper Function definitions
#


def _allowable_batch_size(gpu_memory_frac, row_size):
    free_mem = _device_mem_size(kind="free")
    gpu_memory = free_mem * gpu_memory_frac
    return max(int(gpu_memory / row_size), 1)


def _get_read_engine(engine, file_path, **kwargs):
    LOG.debug("opening '%s' as %s", file_path, engine)
    if engine is None:
        engine = file_path.split(".")[-1]
    if not isinstance(engine, str):
        raise TypeError("Expecting engine as string type.")

    if engine == "csv":
        return CSVFileReader(file_path, **kwargs)
    elif engine == "parquet":
        return PQFileReader(file_path, **kwargs)
    else:
        raise ValueError("Unrecognized read engine.")


#
# GPUFileReader Base Class
#


class GPUFileReader:
    def __init__(
        self, file_path, gpu_memory_frac, batch_size, row_size=None, columns=None, **kwargs
    ):
        """ GPUFileReader Constructor
        """
        self.file_path = file_path
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
            self.file_path
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
        self.file_bytes = os.stat(str(self.file_path)).st_size

        # Use first row to estimate memory-reqs
        names = kwargs.get("names", None)
        dtype = kwargs.get("dtype", None)
        # default csv delim is ","
        sep = kwargs.get("sep", ",")
        self.sep = sep
        self.names = []
        dtype_inf = {}
        nrows = 10
        head = "".join(islice(open(self.file_path), nrows))
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
            free_mem = _device_mem_size(kind="free")
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
                if "hex" in dtype:
                    chunk[col] = chunk[col]._column.nvstrings.htoi()
                    chunk[col] = chunk[col].astype(np.int32)
            else:
                chunk[col] = chunk[col].astype(dtype)


#
# GPUDatasetIterator (Iterates through multiple files)
#


class GPUDatasetIterator:

    """
    Iterates through the files and returns a part of the
    data as a GPU dataframe

    Parameters
    -----------
    paths : list of str
        Path(s) of the data file(s)
    names : list of str
        names of the columns in the dataset
    engine : str
        supported file types are: 'parquet' or 'csv'
    gpu_memory_frac : float
        fraction of the GPU memory to fill
    batch_size : int
        number of samples in each batch
    columns :
    use_row_groups :
    dtypes :
    row_size: int
    """

    def __init__(self, paths, **kwargs):
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            raise TypeError("paths must be a string or a list.")
        if len(paths) < 1:
            raise ValueError("len(paths) must be > 0.")
        self.paths = paths
        self.kwargs = kwargs
        self.cur_path = None

    def __iter__(self):
        for path in self.paths:
            self.cur_path = path
            yield from GPUFileIterator(path, **self.kwargs)


def _shuffle_gdf(gdf, gdf_size=None):
    """ Shuffles a cudf dataframe, returning a new dataframe with randomly
    ordered rows """
    gdf_size = gdf_size or len(gdf)
    arr = cp.arange(gdf_size)
    cp.random.shuffle(arr)
    return gdf.iloc[arr]


class Writer:
    def __init__(self):
        pass

    def add_data(self, gdf):
        raise NotImplementedError()

    def close(self):
        pass


class Shuffler(Writer):
    def __init__(
        self, out_dir, num_out_files=30, num_threads=4, cats=None, conts=None, labels=None
    ):
        self.writer = ParquetWriter(out_dir, num_out_files, num_threads, cats, conts, labels)

    def add_data(self, gdf):
        self.writer.add_data(_shuffle_gdf(gdf))

    def close(self):
        self.writer.close()


class ThreadedWriter(Writer):
    def __init__(
        self, out_dir, num_out_files=30, num_threads=4, cats=None, conts=None, labels=None
    ):
        # set variables
        self.out_dir = out_dir
        self.cats = cats
        self.conts = conts
        self.labels = labels
        self.column_names = None
        if labels and conts:
            self.column_names = labels + conts

        self.num_threads = num_threads
        self.num_out_files = num_out_files
        self.num_samples = [0] * num_out_files

        self.data_files = None

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

    def _write_metadata(self):
        return

    def _write_filelist(self):
        file_list_writer = open(os.path.join(self.out_dir, "file_list.txt"), "w")
        file_list_writer.write(str(self.num_out_files) + "\n")
        for f in self.data_files:
            file_list_writer.write(f + "\n")
        file_list_writer.close()

    def close(self):
        # wake up all the worker threads and signal for them to exit
        for _ in range(self.num_threads):
            self.queue.put(self._eod)

        # wait for pending writes to finish
        self.queue.join()

        self._write_filelist()
        self._write_metadata()

        # Close writers
        for writer in self.data_writers:
            writer.close()


class ParquetWriter(ThreadedWriter):
    def __init__(
        self, out_dir, num_out_files=30, num_threads=4, cats=None, conts=None, labels=None
    ):
        super().__init__(out_dir, num_out_files, num_threads, cats, conts, labels)
        self.data_files = [os.path.join(out_dir, f"{i}.parquet") for i in range(num_out_files)]
        self.data_writers = [pwriter(f, compression=None) for f in self.data_files]

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

    def _write_metadata(self):
        metadata_writer = open(os.path.join(self.out_dir, "metadata.json"), "w")
        data = {}
        data["file_stats"] = []
        for i in range(len(self.data_files)):
            data["file_stats"].append({"file_name": f"{i}.data", "num_rows": self.num_samples[i]})
        data["cats_name"] = self.cats
        data["conts_name"] = self.conts
        data["conts_labels"] = self.labels
        json.dump(data, metadata_writer)
        metadata_writer.close()


class HugeCTRWriter(ThreadedWriter):
    def __init__(
        self, out_dir, num_out_files=30, num_threads=4, cats=None, conts=None, labels=None
    ):
        super().__init__(out_dir, num_out_files, num_threads, cats, conts, labels)
        self.data_files = [os.path.join(out_dir, f"{i}.data") for i in range(num_out_files)]
        self.data_writers = [open(f, "ab") for f in self.data_files]

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

    def _write_metadata(self):
        for i in range(len(self.data_writers)):
            self.data_writers[i].seek(0)
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

            self.data_writers[i].write(header.tobytes())


def _device_mem_size(kind="total"):
    if kind not in ["free", "total"]:
        raise ValueError("kind argument not supported.")
    try:
        if kind == "free":
            return int(cuda.current_context().get_memory_info()[0])
        else:
            return int(cuda.current_context().get_memory_info()[1])
    except NotImplementedError:
        import pynvml

        pynvml.nvmlInit()
        if kind == "free":
            warnings.warn("RMM get_info is not supported. Using total device memory.")
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).total)
        pynvml.nvmlShutdown()
    return size
