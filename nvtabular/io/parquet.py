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
from distutils.version import LooseVersion
from io import BytesIO
from uuid import uuid4

import cudf
import dask
import dask_cudf
from cudf.io.parquet import ParquetWriter as pwriter
from dask.dataframe.io.parquet.utils import _analyze_paths
from dask.utils import natural_sort_key, parse_bytes
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

    def validate_dataset(
        self,
        add_metadata_file=False,
        require_metadata_file=True,
        row_group_max_size=None,
        file_min_size=None,
    ):
        """Validate ParquetDataset object for efficient processing.

        The purpose of this method is to validate that the raw dataset
        meets the minimal requirements for efficient NVTabular processing.
        Warnings are raised if any of the following conditions are not met:

            - The raw dataset directory should contain a global "_metadata"
            file.  If this file is missing, ``add_metadata_file=True`` can
            be passed to generate a new one.
            - If there is no _metadata file, the parquet schema must be
            consistent for all row-groups/files in the raw dataset.
            Otherwise, a new _metadata file must be generated to avoid
            errors at IO time.
            - The row-groups should be no larger than the maximum size limit
            (``row_group_max_size``).
            - For multi-file datasets, the files should be no smaller than
            the minimum size limit (``file_min_size``).

        Parameters
        -----------
        add_metadata_file : bool, default False
            Whether to add a global _metadata file to the dataset if one
            is missing.
        require_metadata_file : bool, default True
            Whether to require the existence of a _metadata file to pass
            the dataset validation.
        row_group_max_size : int or str, default None
            Maximum size (in bytes) of each parquet row-group in the
            dataset. If None, the minimum of ``self.part_size`` and 500MB
            will be used.
        file_min_size : int or str, default None
            Minimum size (in bytes) of each parquet file in the dataset. This
            limit is only applied if there are >1 file in the dataset. If None,
            ``self.part_size`` will be used.

        Returns
        -------
        valid : bool
            Whether or not the input dataset is valid for efficient NVTabular
            processing.
        """

        meta_valid = True  # Parquet format and _metadata exists
        size_valid = False  # Row-group sizes are appropriate

        # Check for user-specified row-group size limit.
        # Otherwise we use the smaller of the dataset partition
        # size and 500MB.
        if row_group_max_size is None:
            row_group_max_size = min(self.part_size, 500_000_000)
        else:
            row_group_max_size = parse_bytes(row_group_max_size)

        # Check for user-specified file size limit.
        # Otherwise we use the smaller of the dataset partition
        # size and 500MB.
        if file_min_size is None:
            file_min_size = self.part_size
        else:
            file_min_size = parse_bytes(file_min_size)

        # Get dataset and path list
        pa_dataset = self._dataset
        paths = [p.path for p in pa_dataset.pieces]
        root_dir, fns = _analyze_paths(paths, self.fs)

        # Collect dataset metadata
        metadata_file_exists = bool(pa_dataset.metadata)
        schema_errors = defaultdict(set)
        if metadata_file_exists:
            # We have a metadata file
            metadata = pa_dataset.metadata
        else:
            # No metadata file - Collect manually
            metadata = None
            for piece, fn in zip(pa_dataset.pieces, fns):
                md = piece.get_metadata()
                md.set_file_path(fn)
                if metadata:
                    _append_row_groups(metadata, md, schema_errors, piece.path)
                else:
                    metadata = md

            # Check for inconsistent schemas.
            # This is not a problem if a _metadata file exists
            for field in schema_errors:
                msg = f"Schema mismatch detected in column: '{field}'."
                warnings.warn(msg)
                for item in schema_errors[field]:
                    msg = f"[{item[0]}] Expected {item[1]}, got {item[2]}."
                    warnings.warn(msg)

            # If there is schema mismatch, urge the user to add a _metadata file
            if len(schema_errors):
                import pyarrow.parquet as pq

                meta_valid = False  # There are schema-mismatch errors

                # Check that the Dask version supports `create_metadata_file`
                if LooseVersion(dask.__version__) < "2.30.0":
                    msg = (
                        "\nThe installed version of Dask is too old to handle "
                        "schema mismatch. Try installing the latest version."
                    )
                    raise warnings.warn(msg)
                    return meta_valid and size_valid  # Early return

                # Collect the metadata with dask_cudf and then convert to pyarrow
                metadata_bytes = dask_cudf.io.parquet.create_metadata_file(
                    paths,
                    out_dir=False,
                )
                with BytesIO() as myio:
                    myio.write(memoryview(metadata_bytes))
                    myio.seek(0)
                    metadata = pq.ParquetFile(myio).metadata

                if not add_metadata_file:
                    msg = (
                        "\nPlease pass add_metadata_file=True to add a global "
                        "_metadata file, or use the regenerate_dataset utility to "
                        "rewrite your dataset. Without a _metadata file, the schema "
                        "mismatch may cause errors at read time."
                    )
                    warnings.warn(msg)

        # Record the total byte size of all row groups and files
        max_rg_size = 0
        max_rg_size_path = None
        file_sizes = defaultdict(int)
        for rg in range(metadata.num_row_groups):
            row_group = metadata.row_group(rg)
            path = row_group.column(0).file_path
            total_byte_size = row_group.total_byte_size
            if total_byte_size > max_rg_size:
                max_rg_size = total_byte_size
                max_rg_size_path = path
            file_sizes[path] += total_byte_size

        # Check if any row groups are prohibitively large.
        # Also check if any row groups are larger than recommended.
        if max_rg_size > row_group_max_size:
            # One or more row-groups are above the "required" limit
            msg = (
                f"Excessive row_group size ({max_rg_size}) detected in file "
                f"{max_rg_size_path}. Please use the regenerate_dataset utility "
                f"to rewrite your dataset."
            )
            warnings.warn(msg)
        else:
            # The only way size_valid==True is if we get here
            size_valid = True

        # Check if any files are smaller than the desired size.
        # We only warn if there are >1 files in the dataset.
        for path, size in file_sizes.items():
            if size < file_min_size and len(pa_dataset.pieces) > 1:
                msg = (
                    f"File {path} is smaller than the desired dataset "
                    f"partition size ({self.part_size}). Consider using the "
                    f"regenerate_dataset utility to rewrite your dataset with a smaller "
                    f"number of (larger) files."
                )
                warnings.warn(msg)
                size_valid = False

        # If the _metadata file is missing, we need to write
        # it (or inform the user that it is missing)
        if not metadata_file_exists:
            if add_metadata_file:
                # Write missing _metadata file
                fs = self.fs
                metadata_path = fs.sep.join([root_dir, "_metadata"])
                with fs.open(metadata_path, "wb") as fil:
                    metadata.write_metadata_file(fil)
                meta_valid = True
            else:
                # Inform user that the _metadata file is missing
                msg = (
                    "For best performance with NVTabular, there should be a "
                    "global _metadata file located in the root directory of the "
                    "dataset. Please pass add_metadata_file=True to add the "
                    "missing file."
                )
                warnings.warn(msg)
                if require_metadata_file:
                    meta_valid = False

        # Return True if we have a parquet dataset with a _metadata file (meta_valid)
        # and the row-groups and file are appropriate sizes (size_valid)
        return meta_valid and size_valid


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


def _append_row_groups(metadata, md, err_collector, path):
    """Helper function to concatenate parquet metadata with
    pyarrow, and catch relevant schema errors.
    """
    try:
        metadata.append_row_groups(md)
    except RuntimeError as err:
        if "requires equal schemas" in str(err):
            schema = metadata.schema.to_arrow_schema()
            schema_new = md.schema.to_arrow_schema()
            for i, name in enumerate(schema.names):
                if schema_new.types[i] != schema.types[i]:
                    err_collector[name].add((path, schema.types[i], schema_new.types[i]))
        else:
            raise err
