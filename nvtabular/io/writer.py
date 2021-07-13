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
import json
import math
import queue
import threading
from typing import Optional

try:
    import cupy as cp
except ImportError:
    cp = None

import numpy as np
from fsspec.core import get_fs_token_paths

from nvtabular.dispatch import annotate

from .shuffle import _shuffle_df


class Writer:
    def add_data(self, df):
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
        cpu=False,
        fns=None,
        suffix=None,
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
        self.fns = [fns] if isinstance(fns, str) else fns
        if self.fns:
            # If sepecific file names were specified,
            # ignore `num_out_files` argument
            self.num_out_files = len(self.fns)
        else:
            self.num_out_files = num_out_files
        self.num_samples = [0] * self.num_out_files

        self.data_paths = None
        self.need_cal_col_names = True
        self.use_guid = use_guid
        self.bytes_io = bytes_io
        self.cpu = cpu
        self.suffix = suffix

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

    def _write_table(self, idx, data):
        return

    def _write_thread(self):
        return

    @annotate("add_data", color="orange", domain="nvt_python")
    def add_data(self, df):
        # Populate columns idxs
        if not self.col_idx:
            _df = df[0] if isinstance(df, list) else df
            for i, x in enumerate(_df.columns.values):
                self.col_idx[str(x)] = i

        if isinstance(df, list):
            # Data is already partitioned into a list
            self._add_data_list(df)
        elif self.num_out_files == 1:
            # Only writing to a single file. No need to
            # scatter or slice the data before writing
            self._add_single_file(df)
        else:
            # Use different mechanism to decompose and write each df
            # partition, depending on the backend (pandas or cudf).
            if self.cpu:
                self._add_data_slice(df)
            else:
                self._add_data_scatter(df)

        # wait for all writes to finish before exiting
        # (so that we aren't using memory)
        if self.num_threads > 1:
            self.queue.join()

    def _add_data_scatter(self, gdf):
        """Write scattered pieces.

        This method is for cudf-backed data only.
        """
        assert not self.cpu

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
            if self.shuffle:
                group = _shuffle_df(group)
            if self.num_threads > 1:
                self.queue.put((x, group))
            else:
                self._write_table(x, group)

    def _add_data_slice(self, df):
        """Write shuffled slices.

        This method works for both pandas and cudf backends.
        """
        # Pandas does not support the `scatter_by_map` method
        # used in `_add_data_scatter`. So, we manually shuffle
        # the df and write out slices.
        if self.shuffle:
            df = _shuffle_df(df)
        int_slice_size = df.shape[0] // self.num_out_files
        slice_size = int_slice_size if df.shape[0] % int_slice_size == 0 else int_slice_size + 1
        for x in range(self.num_out_files):
            start = x * slice_size
            end = start + slice_size
            # check if end is over length
            end = end if end <= df.shape[0] else df.shape[0]
            to_write = df.iloc[start:end]
            self.num_samples[x] = self.num_samples[x] + to_write.shape[0]
            if self.num_threads > 1:
                self.queue.put((x, to_write))
            else:
                self._write_table(x, to_write)

    def _add_data_list(self, dfs):
        """Write a list of DataFrames"""
        assert len(dfs) == self.num_out_files
        for x, df in enumerate(dfs):
            if self.shuffle:
                df = _shuffle_df(df)
            self.num_samples[x] = self.num_samples[x] + df.shape[0]
            if self.num_threads > 1:
                self.queue.put((x, df))
            else:
                self._write_table(x, df)

    def _add_single_file(self, df):
        """Write to single file.

        This method works for both pandas and cudf backends.
        """
        df = _shuffle_df(df) if self.shuffle else df
        self.num_samples[0] = self.num_samples[0] + df.shape[0]
        if self.num_threads > 1:
            self.queue.put((0, df))
        else:
            self._write_table(0, df)

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

    def _close_writers(self) -> Optional[dict]:
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
        _special_meta = self._close_writers()  # pylint: disable=assignment-from-none

        # Move in-memory file to disk
        if self.bytes_io:
            _special_meta = self._bytesio_to_disk()

        return _general_meta, _special_meta

    def _bytesio_to_disk(self):
        raise NotImplementedError("In-memory buffering/shuffling not implemented for this format.")
