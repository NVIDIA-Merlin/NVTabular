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
import functools
import itertools
import logging
import math
import operator
import os
import threading
import warnings
from collections import defaultdict
from distutils.version import LooseVersion
from io import BytesIO
from uuid import uuid4

try:
    import cudf
    import dask_cudf
    from cudf.io.parquet import ParquetWriter as pwriter_cudf
    from dask_cudf.io.parquet import CudfEngine
except ImportError:
    cudf = None
import dask
import dask.dataframe as dd
import fsspec
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
import toolz as tlz
from dask.base import tokenize
from dask.dataframe.core import _concat, new_dd_object
from dask.dataframe.io.parquet.arrow import ArrowDatasetEngine
from dask.dataframe.io.parquet.core import apply_filters
from dask.dataframe.io.parquet.utils import _analyze_paths
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.utils import natural_sort_key, parse_bytes
from fsspec.core import get_fs_token_paths
from pyarrow import parquet as pq
from pyarrow.parquet import ParquetWriter as pwriter_pyarrow

if LooseVersion(dask.__version__) >= "2021.07.1":
    from dask.dataframe.io.parquet.core import aggregate_row_groups
else:
    aggregate_row_groups = None

from ..utils import run_on_worker
from .dataset_engine import DatasetEngine
from .fsspec_utils import _optimized_read_partition_remote, _optimized_read_remote
from .shuffle import Shuffle, _shuffle_df
from .writer import ThreadedWriter

LOG = logging.getLogger("nvtabular")


class CPUParquetEngine(ArrowDatasetEngine):
    @staticmethod
    def read_metadata(*args, **kwargs):
        return _override_read_metadata(ArrowDatasetEngine.read_metadata, *args, **kwargs)

    @classmethod
    def multi_support(cls):
        return hasattr(ArrowDatasetEngine, "multi_support") and ArrowDatasetEngine.multi_support()


# Define GPUParquetEngine if cudf is available
if cudf is not None:

    class GPUParquetEngine(CudfEngine):
        @staticmethod
        def read_metadata(*args, **kwargs):
            if LooseVersion(cudf.__version__).version[:2] == [21, 10]:
                # We only need this work-around for cudf-21.10
                return _override_read_metadata(_cudf_read_metadata, *args, **kwargs)
            return _override_read_metadata(CudfEngine.read_metadata, *args, **kwargs)

        @classmethod
        def multi_support(cls):
            return hasattr(CudfEngine, "multi_support") and CudfEngine.multi_support()

        @classmethod
        def read_partition(cls, fs, pieces, *args, **kwargs):
            if not cudf.utils.ioutils._is_local_filesystem(fs) and (
                not isinstance(pieces, list) or len(pieces) == 1
            ):
                # This version of cudf does not include optimized
                # fsspec usage for remote storage - use custom code path
                return _optimized_read_partition_remote(fs, pieces, *args, **kwargs)
            # Newer versions of cudf are already optimized for s3/gcs
            return CudfEngine.read_partition(fs, pieces, *args, **kwargs)

    def _cudf_read_metadata(*args, **kwargs):
        #
        # NOTE: This function was copied directly from
        #       cudf:branch-21.12 to avoid bugs in
        #       cudf:branch-21.10. This work-around can
        #       be removed when NVTabular is pinned
        #       to cudf>=21.12
        #
        meta, stats, parts, index = ArrowDatasetEngine.read_metadata(*args, **kwargs)
        new_meta = cudf.from_pandas(meta)
        if parts:
            # Re-set "object" dtypes align with pa schema
            set_object_dtypes_from_pa_schema(
                new_meta,
                parts[0].get("common_kwargs", {}).get("schema", None),
            )

        # If `strings_to_categorical==True`, convert objects to int32
        strings_to_cats = kwargs.get("strings_to_categorical", False)
        for col in new_meta._data.names:
            if isinstance(new_meta._data[col], cudf.core.column.StringColumn) and strings_to_cats:
                new_meta._data[col] = new_meta._data[col].astype("int32")
        return (new_meta, stats, parts, index)

    def set_object_dtypes_from_pa_schema(df, schema):
        #
        # NOTE: This utility was copied directly from
        #       cudf:branch-21.12 to avoid bugs in
        #       cudf:branch-21.10. This work-around can
        #       be removed when NVTabular is pinned
        #       to cudf>=21.12
        #
        # Simple utility to modify cudf DataFrame
        # "object" dtypes to agree with a specific
        # pyarrow schema
        if schema:
            for col_name, col in df._data.items():
                if col_name is None:
                    # Pyarrow cannot handle `None` as a field name.
                    # However, this should be a simple range index that
                    # we can ignore anyway
                    continue
                typ = cudf.utils.dtypes.cudf_dtype_from_pa_type(schema.field(col_name).type)
                if (
                    col_name in schema.names
                    and not isinstance(typ, (cudf.ListDtype, cudf.StructDtype))
                    and isinstance(col, cudf.core.column.StringColumn)
                ):
                    df._data[col_name] = col.astype(typ)


def _override_read_metadata(
    parent_read_metadata,
    fs,
    paths,
    index=None,
    gather_statistics=None,
    split_row_groups=None,
    filters=None,
    aggregate_files=None,
    dataset=None,
    chunksize=None,
    **global_kwargs,
):
    # This function is used by both CPU and GPU-backed
    # ParquetDatasetEngine instances to override the `read_metadata`
    # component of the upstream `read_parquet` logic. This provides
    # NVTabular with direct access to the final partitioning behavior.

    # For now, disallow the user from setting `chunksize`
    if chunksize:
        raise ValueError(
            "NVTabular does not yet support the explicit use " "of Dask's `chunksize` argument."
        )

    # Extract metadata_collector from the dataset "container"
    dataset = dataset or {}
    metadata_collector = dataset.pop("metadata_collector", None)

    # Gather statistics by default.
    # This enables optimized length calculations
    if gather_statistics is None:
        gather_statistics = True

    # Use a local_kwarg dictionary to make it easier to exclude
    # `aggregate_files` for older Dask versions
    local_kwargs = {
        "index": index,
        "filters": filters,
        # Use chunksize=1 to "ensure" statistics are gathered
        # if `gather_statistics=True`.  Note that Dask will bail
        # from statistics gathering if it does not expect statistics
        # to be "used" after `read_metadata` returns.
        "chunksize": 1 if gather_statistics else None,
        "gather_statistics": gather_statistics,
        "split_row_groups": split_row_groups,
    }
    if aggregate_row_groups is not None:
        # File aggregation is only available for Dask>=2021.07.1
        local_kwargs["aggregate_files"] = aggregate_files
    elif aggregate_files:
        raise ValueError("This version of Dask does not support the `aggregate_files` argument.")

    # Start with "super-class" read_metadata logic
    read_metadata_result = parent_read_metadata(
        fs,
        paths,
        **local_kwargs,
        **global_kwargs,
    )
    parts = read_metadata_result[2].copy()
    statistics = read_metadata_result[1].copy()

    # Process the statistics.
    # Note that these steps are usaually performed after
    # `engine.read_metadata` returns (in Dask), but we are doing
    # it ourselves in NVTabular (to capture the expected output
    # partitioning plan)
    if statistics:
        result = list(
            zip(*[(part, stats) for part, stats in zip(parts, statistics) if stats["num-rows"] > 0])
        )
        parts, statistics = result or [[], []]

        # Apply filters
        if filters:
            parts, statistics = apply_filters(parts, statistics, filters)

        # Apply file aggregation
        if aggregate_row_groups is not None:

            # Convert `aggregate_files` to an integer `aggregation_depth`
            aggregation_depth = False
            if len(parts) and aggregate_files:
                aggregation_depth = parts[0].get("aggregation_depth", aggregation_depth)

            # Aggregate parts/statistics if we are splitting by row-group
            if chunksize or (split_row_groups and int(split_row_groups) > 1):
                parts, statistics = aggregate_row_groups(
                    parts, statistics, chunksize, split_row_groups, fs, aggregation_depth
                )

    # Update `metadata_collector` and return the "original" `read_metadata_result`
    metadata_collector["stats"] = statistics
    metadata_collector["parts"] = parts
    return read_metadata_result


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
        cpu=False,
        **kwargs,
    ):
        super().__init__(paths, part_size, cpu=cpu, storage_options=storage_options)
        self._pp_map = None
        self._pp_nrows = None
        self._pp_metadata = None
        self._real_meta = None

        # Process `kwargs`
        self.read_parquet_kwargs = kwargs.copy()
        self.aggregate_files = self.read_parquet_kwargs.pop("aggregate_files", False)
        self.filters = self.read_parquet_kwargs.pop("filters", None)
        self.dataset_kwargs = self.read_parquet_kwargs.pop("dataset", {})

        if row_groups_per_part is None:
            self._real_meta, rg_byte_size_0 = run_on_worker(
                _sample_row_group, self._path0, self.fs, cpu=self.cpu, memory_usage=True
            )
            row_groups_per_part = self.part_size / rg_byte_size_0
            if row_groups_per_part < 1.0:
                warnings.warn(
                    f"Row group memory size ({rg_byte_size_0}) (bytes) of parquet file is bigger"
                    f" than requested part_size ({self.part_size}) for the NVTabular dataset."
                    f"A row group memory size of 128 MB is generally recommended. You can find"
                    f" info on how to set the row group size of parquet files in "
                    f"https://nvidia-merlin.github.io/NVTabular/main/resources/troubleshooting.html"
                    f"#setting-the-row-group-size-for-the-parquet-files"
                )
                row_groups_per_part = 1.0

        self.row_groups_per_part = int(row_groups_per_part)

        assert self.row_groups_per_part > 0

    @property
    @functools.lru_cache(1)
    def _path0(self):
        return next(self._dataset.get_fragments()).path

    @property
    @functools.lru_cache(1)
    def _legacy_dataset(self):
        # TODO: Remove this after finding a way to avoid
        # the use of `ParquetDataset` in `validate_dataset`
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
    def _dataset(self):
        paths = self.stripped_paths
        fs = self.fs
        if len(paths) > 1:
            # This is a list of files
            dataset = pa_ds.dataset(paths, filesystem=fs)
        else:
            # This is a directory or a single file
            dataset = pa_ds.dataset(paths[0], filesystem=fs)
        return dataset

    @property
    def _file_partition_map(self):
        if self._pp_map is None:
            self._process_parquet_metadata()
        return self._pp_map

    @property
    def _partition_lens(self):
        if self._pp_nrows is None:
            self._process_parquet_metadata()
        return self._pp_nrows

    @property
    def num_rows(self):
        # TODO: Avoid parsing metadata once upstream dask
        # can get the length efficiently (in all practical cases)
        if self._partition_lens:
            return sum(self._partition_lens)
        return len(self.to_ddf().index)

    def _process_parquet_metadata(self):
        # Utility shared by `_file_partition_map` and `_partition_lens`
        # to collect useful information from the parquet metadata

        # First, we need to populate `self._pp_metadata`
        if self._pp_metadata is None:
            _ = self.to_ddf()

        # Second, we can use the path and num-rows information
        # in parts and stats
        parts = self._pp_metadata["parts"]
        stats = self._pp_metadata["stats"]
        _pp_map = {}
        _pp_nrows = []
        distinct_files = True
        for i, (part, stat) in enumerate(zip(parts, stats)):
            if distinct_files:
                if isinstance(part, list):
                    if len(part) > 1:
                        distinct_files = False
                    else:
                        path = part[0]["piece"][0]
                        _pp_map[path] = i
                else:
                    path = part["piece"][0]
                    _pp_map[path] = i
            _pp_nrows.append(stat["num-rows"])

        self._pp_nrows = _pp_nrows
        self._pp_map = _pp_map

    def to_ddf(self, columns=None, cpu=None):

        # Check if we are using cpu or gpu backend
        cpu = self.cpu if cpu is None else cpu
        backend_engine = CPUParquetEngine if cpu else GPUParquetEngine

        # Use dask-dataframe with appropriate engine
        metadata_collector = {"stats": [], "parts": []}
        dataset_kwargs = {"metadata_collector": metadata_collector}
        dataset_kwargs.update(self.dataset_kwargs)
        ddf = dd.read_parquet(
            self.paths,
            columns=columns,
            engine=backend_engine,
            index=False,
            aggregate_files=self.aggregate_files,
            filters=self.filters,
            split_row_groups=self.row_groups_per_part,
            storage_options=self.storage_options,
            dataset=dataset_kwargs,
            **self.read_parquet_kwargs,
        )
        self._pp_metadata = metadata_collector
        return ddf

    def to_cpu(self):
        self.cpu = True

    def to_gpu(self):
        self.cpu = False

    def sample_data(self, n=1):
        """Return a real data sample from the Dataset"""
        if self._real_meta is not None:
            # First check is we already cached a data sample
            # while calculating `row_groups_per_part`
            _len = len(self._real_meta)
            if _len >= n:
                if _len == n:
                    _real_meta = self._real_meta
                else:
                    _real_meta = self._real_meta.take(list(range(n)))
                # We can clear self._real_meta, because the data
                # will still be cached at the Dataset level
                self._real_meta = None
                return _real_meta

        # Real metadata sample is not cached - Sample from
        # the first row-group in the Dataset
        return run_on_worker(
            _sample_row_group,
            self._path0,
            self.fs,
            cpu=self.cpu,
            n=n,
            memory_usage=False,
        ).take(list(range(n)))

    def validate_dataset(
        self,
        add_metadata_file=False,
        require_metadata_file=True,
        row_group_max_size=None,
        file_min_size=None,
    ):
        """Validate ParquetDatasetEngine object for efficient processing.

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
        pa_dataset = self._legacy_dataset
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
                meta_valid = False  # There are schema-mismatch errors

                # Check that the Dask version supports `create_metadata_file`
                if LooseVersion(dask.__version__) <= "2.30.0":
                    msg = (
                        "\nThe installed version of Dask is too old to handle "
                        "schema mismatch. Try installing the latest version."
                    )
                    warnings.warn(msg)
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

    @classmethod
    def regenerate_dataset(
        cls,
        dataset,
        output_path,
        columns=None,
        file_size=None,
        part_size=None,
        cats=None,
        conts=None,
        labels=None,
        storage_options=None,
    ):
        """Regenerate an NVTabular Dataset for efficient processing.

        Example Usage::

            dataset = Dataset("/path/to/data_pq", engine="parquet")
            dataset.regenerate_dataset(
                out_path, part_size="1MiB", file_size="10MiB"
            )

        Parameters
        -----------
        dataset : Dataset
            Input `Dataset` object (to be regenerated).
        output_path : string
            Root directory path to use for the new (regenerated) dataset.
        columns : list[string], optional
            Subset of columns to include in the regenerated dataset.
        file_size : int or string, optional
            Desired size of each output file.
        part_size : int or string, optional
            Desired partition size to use within regeneration algorithm.
            Note that this is effectively the size of each contiguous write
            operation in cudf.
        cats : list[string], optional
            Categorical column list.
        conts : list[string], optional
            Continuous column list.
        labels : list[string], optional
            Label column list.
        storage_options : dict, optional
            Storage-option kwargs to pass through to the `fsspec` file-system
            interface.

        Returns
        -------
        result : int or Delayed
            If `compute=True` (default), the return value will be an integer
            corresponding to the number of generated data files.  If `False`,
            the returned value will be a `Delayed` object.
        """

        # Specify ideal file size and partition size
        row_group_size = 128_000_000
        file_size = parse_bytes(file_size) or row_group_size * 100
        part_size = parse_bytes(part_size) or row_group_size * 10
        part_size = min(part_size, file_size)

        fs, _, _ = get_fs_token_paths(output_path, mode="wb", storage_options=storage_options)

        # Start by converting the original dataset to a Dask-Dataframe
        # object in CPU memory.  We avoid GPU memory in case the original
        # dataset is prone to OOM errors.
        _ddf = dataset.engine.to_ddf(columns=columns, cpu=True)

        # Prepare general metadata (gmd)
        gmd = {}
        cats = cats or []
        conts = conts or []
        labels = labels or []
        if not len(cats + conts + labels):
            warnings.warn(
                "General-metadata information not detected! "
                "Please pass lists for `cats`, `conts`, and `labels` as"
                "arguments to `regenerate_dataset` to ensure a complete "
                "and correct _metadata.json file."
            )
        col_idx = {str(name): i for i, name in enumerate(_ddf.columns)}
        gmd["cats"] = [{"col_name": c, "index": col_idx[c]} for c in cats]
        gmd["conts"] = [{"col_name": c, "index": col_idx[c]} for c in conts]
        gmd["labels"] = [{"col_name": c, "index": col_idx[c]} for c in labels]

        # Get list of partition lengths
        token = tokenize(
            dataset,
            output_path,
            columns,
            part_size,
            file_size,
            cats,
            conts,
            labels,
            storage_options,
        )
        getlen_name = "getlen-" + token
        name = "all-" + getlen_name
        dsk = {(getlen_name, i): (len, (_ddf._name, i)) for i in range(_ddf.npartitions)}
        dsk[name] = [(getlen_name, i) for i in range(_ddf.npartitions)]
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[_ddf])
        size_list = Delayed(name, graph).compute()

        # Get memory usage per row using first partition
        p0_mem_size = _ddf.partitions[0].memory_usage(deep=True, index=True).sum().compute()
        mem_per_row = int(float(p0_mem_size) / float(size_list[0]))

        # Determine the number of rows to assign to each output partition
        # and the number of output partitions to assign to each output file
        rows_per_part = int(part_size / mem_per_row)
        parts_per_file = int(file_size / part_size)

        # Construct re-partition graph
        dsk2 = {}
        repartition_name = "repartition-" + token
        split_name = "split-" + repartition_name
        getitem_name = "getitem-" + repartition_name

        gets = defaultdict(list)
        out_parts = 0
        remaining_out_part_rows = rows_per_part
        for i, in_part_size in enumerate(size_list):

            # The `split` dictionary will be passed to this input
            # partition to dictate how that partition will be split
            # into different output partitions/files.  The "key" of
            # this dict is the output partition, and the value is a
            # tuple specifying the (start, end) row range.
            split = {}
            last = 0
            while in_part_size >= remaining_out_part_rows:

                gets[out_parts].append(i)
                split[out_parts] = (last, last + remaining_out_part_rows)
                last += remaining_out_part_rows
                in_part_size = in_part_size - remaining_out_part_rows

                remaining_out_part_rows = rows_per_part
                out_parts += 1

            if in_part_size:
                gets[out_parts].append(i)
                split[out_parts] = (last, last + in_part_size)
                remaining_out_part_rows -= in_part_size

            if remaining_out_part_rows == 0:
                remaining_out_part_rows = rows_per_part
                out_parts += 1

            dsk2[(split_name, i)] = (_split_part, (_ddf._name, i), split)
        npartitions = max(gets) + 1

        for k, v_list in gets.items():
            last = None
            _concat_list = []
            for v in v_list:
                key = (getitem_name, v, k)
                _concat_list.append(key)
                dsk2[key] = (operator.getitem, (split_name, v), k)

            ignore_index = True
            dsk2[(repartition_name, k)] = (_concat, _concat_list, ignore_index)

        graph2 = HighLevelGraph.from_collections(repartition_name, dsk2, dependencies=[_ddf])
        divisions = [None] * (npartitions + 1)
        _ddf2 = new_dd_object(graph2, repartition_name, _ddf._meta, divisions)

        # Make sure the root directory exists
        fs.mkdirs(output_path, exist_ok=True)

        # Construct rewrite graph
        dsk3 = {}
        rewrite_name = "rewrite-" + token
        write_data_name = "write-data-" + rewrite_name
        write_metadata_name = "write-metadata-" + rewrite_name
        inputs = []
        final_inputs = []
        for i in range(_ddf2.npartitions):
            index = i // parts_per_file
            nex_index = (i + 1) // parts_per_file
            package_task = (index != nex_index) or (i == (_ddf2.npartitions - 1))
            fn = f"part.{index}.parquet"
            inputs.append((repartition_name, i))
            if package_task:
                final_inputs.append((write_data_name, i))
                dsk3[(write_data_name, i)] = (
                    _write_data,
                    inputs,
                    output_path,
                    fs,
                    fn,
                )
                inputs = []

        # Final task collects and writes all metadata
        dsk3[write_metadata_name] = (
            _write_metadata_file,
            final_inputs,
            fs,
            output_path,
            gmd,
        )
        graph3 = HighLevelGraph.from_collections(write_metadata_name, dsk3, dependencies=[_ddf2])

        return Delayed(write_metadata_name, graph3)


def _write_metadata_file(md_list, fs, output_path, gmd_base):

    # Prepare both "general" and parquet metadata
    gmd = gmd_base.copy()
    pmd = {}
    data_paths = []
    file_stats = []
    for m in md_list:
        for path in m.keys():
            md = m[path]["md"]
            rows = m[path]["rows"]
            pmd[path] = md
            data_paths.append(path)
            fn = path.split(fs.sep)[-1]
            file_stats.append({"file_name": fn, "num_rows": rows})
    gmd["data_paths"] = data_paths
    gmd["file_stats"] = file_stats

    # Write general metadata file
    GPUParquetWriter.write_general_metadata(gmd, fs, output_path)

    # Write specialized parquet metadata file
    GPUParquetWriter.write_special_metadata(pmd, fs, output_path)

    # Return total file count (sanity check)
    return len(data_paths)


def _write_data(data_list, output_path, fs, fn):

    # Initialize chunked writer
    path = fs.sep.join([output_path, fn])
    writer = pwriter_cudf(path, compression=None)
    rows = 0

    # Loop over the data_list, convert to cudf,
    # and append to the file
    for data in data_list:
        rows += len(data)
        writer.write_table(cudf.from_pandas(data))

    # Return metadata and row-count in dict
    return {fn: {"md": writer.close(metadata_file_path=fn), "rows": rows}}


class BaseParquetWriter(ThreadedWriter):
    def __init__(self, out_dir, suffix=".parquet", **kwargs):
        super().__init__(out_dir, **kwargs)
        self.data_paths = []
        self.data_files = []
        self.data_writers = []
        self.data_bios = []
        self._lock = threading.RLock()
        self.pwriter = self._pwriter
        self.pwriter_kwargs = {}
        self.suffix = suffix

    @property
    def _pwriter(self):
        """Returns ParquetWriter Backend Class"""
        raise (NotImplementedError)

    def _read_parquet(self, source):
        """Read parquet data from source"""
        raise (NotImplementedError)

    def _to_parquet(self, df, sink):
        """Write data to parquet and return pq metadata"""
        raise (NotImplementedError)

    def _get_filename(self, i):
        if self.fns:
            fn = self.fns[i]
        elif self.use_guid:
            fn = f"{i}.{guid()}{self.suffix}"
        else:
            fn = f"{i}{self.suffix}"

        return os.path.join(self.out_dir, fn)

    def _append_writer(self, path, schema=None, add_args=None, add_kwargs=None):
        # Add additional args and kwargs
        _args = add_args or []
        _kwargs = tlz.merge(self.pwriter_kwargs, add_kwargs or {})

        if self.bytes_io:
            bio = BytesIO()
            self.data_bios.append(bio)
            self.data_writers.append(self.pwriter(bio, *_args, **_kwargs))
        else:
            f = fsspec.open(path, mode="wb").open()
            self.data_files.append(f)
            self.data_writers.append(self.pwriter(f, *_args, **_kwargs))

    def _get_or_create_writer(self, idx, schema=None):
        # lazily initializes a writer for the given index
        with self._lock:
            while len(self.data_writers) <= idx:
                # Append writer
                path = self._get_filename(len(self.data_writers))
                self.data_paths.append(path)
                self._append_writer(path, schema=schema)
            return self.data_writers[idx]

    def _write_table(self, idx, data):
        """Write data"""
        raise (NotImplementedError)

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
        """Write global _metadata file"""
        raise (NotImplementedError)

    def _close_writers(self):
        """Close writers and return extracted metadata"""
        raise (NotImplementedError)

    def _bytesio_to_disk(self):
        md = {}
        for bio, path in zip(self.data_bios, self.data_paths):
            df = self._read_parquet(bio)
            bio.close()
            if self.shuffle == Shuffle.PER_WORKER:
                df = _shuffle_df(df)
            md[path] = self._to_parquet(df, path)
        return md


class GPUParquetWriter(BaseParquetWriter):
    def __init__(self, out_dir, **kwargs):
        super().__init__(out_dir, **kwargs)
        # Passing index=False when creating ParquetWriter
        # to avoid bug: https://github.com/rapidsai/cudf/issues/7011
        self.pwriter_kwargs = {"compression": None, "index": False}

    @property
    def _pwriter(self):
        return pwriter_cudf

    def _read_parquet(self, source):
        return cudf.io.read_parquet(source)

    def _to_parquet(self, df, sink):
        fn = sink.split(self.fs.sep)[-1]
        return df.to_parquet(sink, metadata_file_path=fn, compression=None, index=False)

    def _write_table(self, idx, data):
        writer = self._get_or_create_writer(idx)
        writer.write_table(data)

    @classmethod
    def write_special_metadata(cls, md, fs, out_dir):
        # Sort metadata by file name and convert list of
        # tuples to a list of metadata byte-blobs
        md_list = [m[1] for m in sorted(list(md.items()), key=lambda x: natural_sort_key(x[0]))]

        # Aggregate metadata and write _metadata file
        _write_pq_metadata_file_cudf(md_list, fs, out_dir)

    def _close_writers(self):
        md_dict = {}
        _fns = self.fns or [path.split(self.fs.sep)[-1] for path in self.data_paths]
        for writer, fn in zip(self.data_writers, _fns):
            md_dict[fn] = writer.close(metadata_file_path=fn)
        for f in self.data_files:
            f.close()
        return md_dict


class CPUParquetWriter(BaseParquetWriter):
    def __init__(self, out_dir, **kwargs):
        super().__init__(out_dir, **kwargs)
        self.md_collectors = {}
        self.pwriter_kwargs = {"compression": None}

    @property
    def _pwriter(self):
        return pwriter_pyarrow

    def _read_parquet(self, source):
        return pd.read_parquet(source, engine="pyarrow")

    def _get_row_group_size(self, df):
        # Make sure our `row_group_size` argument (which corresponds
        # to the number of rows in each row-group) will produce
        # row-groups ~128MB in size.
        if not hasattr(self, "_row_group_size"):
            row_size = df.memory_usage(deep=True).sum() / max(len(df), 1)
            self._row_group_size = math.ceil(128_000_000 / row_size)
        return self._row_group_size

    def _to_parquet(self, df, sink):
        md = []
        df.to_parquet(
            sink,
            row_group_size=self._get_row_group_size(df),
            metadata_collector=md,
            compression=None,
            index=False,
        )
        fn = sink.split(self.fs.sep)[-1]
        md[0].set_file_path(fn)
        return md

    def _append_writer(self, path, schema=None):

        # Define "metadata collector" for pyarrow
        _md_collector = []
        _args = [schema]
        _kwargs = {"metadata_collector": _md_collector}

        # Use `BaseParquetWriter` logic
        super()._append_writer(path, add_args=_args, add_kwargs=_kwargs)

        # Keep track of "metadata collector" for pyarrow
        self.md_collectors[path] = _md_collector

    def _write_table(self, idx, data):
        table = pa.Table.from_pandas(data, preserve_index=False)
        writer = self._get_or_create_writer(idx, schema=table.schema)
        writer.write_table(table, row_group_size=self._get_row_group_size(data))

    @classmethod
    def write_special_metadata(cls, md, fs, out_dir):
        # Sort metadata by file name and convert list of
        # tuples to a list of metadata byte-blobs
        md_list = [m[1] for m in sorted(list(md.items()), key=lambda x: natural_sort_key(x[0]))]

        # Aggregate metadata and write _metadata file
        _write_pq_metadata_file_pyarrow(md_list, fs, out_dir)

    def _close_writers(self):
        _fns = self.fns or [path.split(self.fs.sep)[-1] for path in self.data_paths]
        for writer, fn in zip(self.data_writers, _fns):
            writer.close()
            _path = self.fs.sep.join([str(self.out_dir), fn])
            if _path in self.md_collectors:
                self.md_collectors[_path][0].set_file_path(fn)
        return self.md_collectors


def _write_pq_metadata_file_cudf(md_list, fs, path):
    """Converts list of parquet metadata objects into a single shared _metadata file."""
    if md_list:
        metadata_path = fs.sep.join([path, "_metadata"])
        _meta = cudf.io.merge_parquet_filemetadata(md_list) if len(md_list) > 1 else md_list[0]
        with fs.open(metadata_path, "wb") as fil:
            fil.write(bytes(_meta))
    return


def _write_pq_metadata_file_pyarrow(md_list, fs, path):
    """Converts list of parquet metadata objects into a single shared _metadata file."""
    if md_list:
        metadata_path = fs.sep.join([path, "_metadata"])
        _meta = None
        for md in itertools.chain(*md_list):
            if _meta is None:
                _meta = md
            else:
                _meta.append_row_groups(md)
        with fs.open(metadata_path, "wb") as fil:
            _meta.write_metadata_file(fil)
    return


def guid():
    """Simple utility function to get random hex string"""
    return uuid4().hex


def _memory_usage(df):
    """Return the total memory usage of a DataFrame"""
    return df.memory_usage(deep=True).sum()


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


def _split_part(x, split):
    out = {}
    for k, v in split.items():
        out[k] = x.iloc[v[0] : v[1]]
    return out


def _sample_row_group(path, fs, cpu=False, n=1, memory_usage=False):
    """Return the first Parquet Row-Group for a given path

    The memory_usage of the row-group will also be returned
    if `memory_usage=True`.
    """
    if cpu:
        with fs.open(path, "rb") as f0:
            # Use pyarrow for CPU version.
            # Pandas does not enable single-row-group access.
            _df = pq.ParquetFile(f0).read_row_group(0).to_pandas()
    else:
        if cudf.utils.ioutils._is_local_filesystem(fs):
            # Allow cudf to open the file if this is a local file
            # system (can be significantly faster in this case)
            _df = cudf.io.read_parquet(path, row_groups=0)
        else:
            _df = _optimized_read_remote(path, 0, None, fs)
    _indices = list(range(n))
    if memory_usage:
        return _df.take(_indices), _memory_usage(_df)
    return _df.take(_indices)
