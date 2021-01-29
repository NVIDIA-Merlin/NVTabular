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
import json
import math
import queue
import threading

import cupy as cp
import numpy as np
from cudf.utils.dtypes import is_list_dtype
from fsspec.core import get_fs_token_paths
from nvtx import annotate


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

        # Only use threading if num_threads > 1
        self.queue = None
        if self.num_threads > 1:
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

    def _write_table(self, idx, data, has_list_column=False):
        return

    def _write_thread(self):
        return

    @annotate("add_data", color="orange", domain="nvt_python")
    def add_data(self, gdf):
        # Populate columns idxs
        if not self.col_idx:
            for i, x in enumerate(gdf.columns.values):
                self.col_idx[str(x)] = i

        # list columns in cudf don't currently support chunked writing in parquet.
        # hack around this by just writing a single file with this partition
        # this restriction can be removed once cudf supports chunked writing
        # in parquet
        if any(is_list_dtype(gdf[col].dtype) for col in gdf.columns):
            self._write_table(0, gdf, True)
            return

        # Generate `ind` array to map each row to an output file.
        # This approach is certainly more optimized for shuffling
        # than it is for non-shuffling, but using a single code
        # path is probably worth the (possible) minor overhead.
        nrows = gdf.shape[0]
        typ = np.min_scalar_type(nrows * 2)
        if self.shuffle:
            ind = cp.random.choice(cp.arange(self.num_out_files, dtype=typ), nrows)
        else:
            ind = cp.arange(nrows, dtype=typ)
            cp.floor_divide(ind, math.ceil(nrows / self.num_out_files), out=ind)
        for x, group in enumerate(
            gdf.scatter_by_map(ind, map_size=self.num_out_files, keep_index=False)
        ):
            self.num_samples[x] += len(group)
            if self.num_threads > 1:
                self.queue.put((x, group))
            else:
                self._write_table(x, group)

        # wait for all writes to finish before exiting
        # (so that we aren't using memory)
        if self.num_threads > 1:
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
        if self.num_threads > 1:
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
