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
import functools
import logging
import os
import threading
import warnings
from collections import defaultdict
from io import BytesIO
from uuid import uuid4

import cudf
from cudf.io.parquet import ParquetWriter as pwriter
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.dataframe.io.parquet.utils import _analyze_paths
from dask.utils import natural_sort_key
from nvtx import annotate
from pyarrow import parquet as pq

from .dataset_engine import DatasetEngine
from .shuffle import Shuffle, _shuffle_gdf
from .writer import ThreadedWriter

LOG = logging.getLogger("nvtabular")


class ParquetDatasetEngine(DatasetEngine):
    """ParquetDatasetEngine is a Dask-based version of cudf.read_parquet."""

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
            rg_byte_size_0 = _memory_usage(cudf.io.read_parquet(path0, row_groups=0, row_group=0))
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


class ParquetWriter(ThreadedWriter):
    def __init__(self, out_dir, **kwargs):
        super().__init__(out_dir, **kwargs)
        self.data_paths = []
        self.data_writers = []
        self.data_bios = []
        self._lock = threading.RLock()

    def _get_filename(self, i):
        if self.use_guid:
            fn = f"{i}.{guid()}.parquet"
        else:
            fn = f"{i}.parquet"

        return os.path.join(self.out_dir, fn)

    def _get_or_create_writer(self, idx):
        # lazily initializes a writer for the given index
        with self._lock:
            while len(self.data_writers) <= idx:
                path = self._get_filename(len(self.data_writers))
                self.data_paths.append(path)
                if self.bytes_io:
                    bio = BytesIO()
                    self.data_bios.append(bio)
                    self.data_writers.append(pwriter(bio, compression=None))
                else:
                    self.data_writers.append(pwriter(path, compression=None))

            return self.data_writers[idx]

    def _write_table(self, idx, data, has_list_column=False):
        if has_list_column:
            # currently cudf doesn't support chunked parquet writers with list columns
            # write out a new file, rather than stream multiple chunks to a single file
            filename = self._get_filename(len(self.data_paths))
            data.to_parquet(filename)
            self.data_paths.append(filename)
        else:
            writer = self._get_or_create_writer(idx)
            writer.write_table(data)

    def _write_thread(self):
        while True:
            item = self.queue.get()
            try:
                if item is self._eod:
                    break
                idx, data = item
                with self.write_locks[idx]:
                    self._write_table(idx, data, False)
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
            if self.shuffle == Shuffle.PER_WORKER:
                gdf = _shuffle_gdf(gdf)
            gdf.to_parquet(path, compression=None, index=False)
        return


def _write_pq_metadata_file(md_list, fs, path):
    """ Converts list of parquet metadata objects into a single shared _metadata file. """
    if md_list:
        metadata_path = fs.sep.join([path, "_metadata"])
        _meta = cudf.io.merge_parquet_filemetadata(md_list) if len(md_list) > 1 else md_list[0]
        with fs.open(metadata_path, "wb") as fil:
            _meta.tofile(fil)
    return


def guid():
    """Simple utility function to get random hex string"""
    return uuid4().hex


def _memory_usage(df):
    """this function is a workaround of a problem with getting memory usage of lists
    in cudf0.16.  This can be deleted and just use `df.memory_usage(deep= True, index=True).sum()`
    once we are using cudf 0.17 (fixed in https://github.com/rapidsai/cudf/pull/6549)"""
    size = 0
    for col in df._data.columns:
        if cudf.utils.dtypes.is_list_dtype(col.dtype):
            for child in col.base_children:
                size += child.__sizeof__()
        else:
            size += col._memory_usage(deep=True)
    size += df.index.memory_usage(deep=True)
    return size
