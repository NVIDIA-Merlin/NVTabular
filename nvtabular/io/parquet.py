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
from io import BytesIO
from uuid import uuid4

import cudf
import dask_cudf
from cudf.io.parquet import ParquetWriter as pwriter
from dask.utils import natural_sort_key
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
        batch_size=None,  # Ignored
    ):
        super().__init__(paths, part_size, storage_options)
        if row_groups_per_part is None:
            path0 = self._dataset.pieces[0].path
            rg_byte_size_0 = _memory_usage(cudf.io.read_parquet(path0, row_groups=0, row_group=0))
            row_groups_per_part = self.part_size / rg_byte_size_0
            if row_groups_per_part < 1.0:
                warnings.warn(
                    f"Row group memory size ({rg_byte_size_0}) (bytes) of parquet file is bigger"
                    f" than requested part_size ({self.part_size}) for the NVTabular dataset."
                    f"A row group memory size of 128 MB is generally recommended. You can find"
                    f" info on how to set the row group size of parquet files in "
                    f"https://nvidia.github.io/NVTabular/main/HowItWorks.html"
                    f"#getting-your-data-ready-for-nvtabular"
                )
                row_groups_per_part = 1.0

        self.row_groups_per_part = int(row_groups_per_part)

        assert self.row_groups_per_part > 0

    @property
    @functools.lru_cache(1)
    def _dataset(self):
        paths = self.paths
        fs = self.fs
        if len(paths) > 1:
            # This is a list of files
            dataset = pq.ParquetDataset(paths, filesystem=fs, validate_schema=False)
        elif fs.isdir(paths[0]):
            # This is a directory
            dataset = pq.ParquetDataset(paths[0], filesystem=fs, validate_schema=False)
        else:
            # This is a single file
            dataset = pq.ParquetDataset(paths[0], filesystem=fs)
        return dataset

    @property
    @functools.lru_cache(1)
    def num_rows(self):
        # TODO: Avoid parsing metadata here if we can confirm upstream dask
        # can get the length efficiently (in all practical cases)
        dataset = self._dataset
        if dataset.metadata:
            # We have a metadata file
            return dataset.metadata.num_rows
        else:
            # Sum up row-group sizes manually
            num_rows = 0
            for piece in dataset.pieces:
                num_rows += piece.get_metadata().num_rows
            return num_rows

    def to_ddf(self, columns=None):
        return dask_cudf.read_parquet(
            self.paths,
            columns=columns,
            # can't omit reading the index in if we aren't being passed columns
            index=None if columns is None else False,
            gather_statistics=False,
            split_row_groups=self.row_groups_per_part,
            storage_options=self.storage_options,
        )


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
