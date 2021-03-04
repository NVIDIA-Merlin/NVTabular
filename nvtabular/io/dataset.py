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

import logging
import random
import warnings

import cudf
import dask
import dask_cudf
import numpy as np
import pandas as pd
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path

from nvtabular.io.shuffle import _check_shuffle_arg

from ..utils import device_mem_size
from .csv import CSVDatasetEngine
from .dask import _ddf_to_dataset
from .dataframe_engine import DataFrameDatasetEngine
from .parquet import ParquetDatasetEngine

LOG = logging.getLogger("nvtabular")


class Dataset:
    """Universal external-data wrapper for NVTabular

    The NVTabular `Workflow` and `DataLoader`-related APIs require all
    external data to be converted to the universal `Dataset` type.  The
    main purpose of this class is to abstract away the raw format of the
    data, and to allow other NVTabular classes to reliably materialize a
    `dask_cudf.DataFrame` collection (and/or collection-based iterator)
    on demand.

    A new `Dataset` object can be initialized from a variety of different
    raw-data formats. To initialize an object from a directory path or
    file list, the `engine` argument should be used to specify either
    "parquet" or "csv" format.  If the first argument contains a list
    of files with a suffix of either "parquet" or "csv", the engine can
    be inferred::

        # Initialize Dataset with a parquet-dataset directory.
        # must specify engine="parquet"
        dataset = Dataset("/path/to/data_pq", engine="parquet")

        # Initialize Dataset with list of csv files.
        # engine="csv" argument is optional
        dataset = Dataset(["file_0.csv", "file_1.csv"])

    Since NVTabular leverages `fsspec` as a file-system interface,
    the underlying data can be stored either locally, or in a remote/cloud
    data store.  To read from remote storage, like gds or s3, the
    appropriate protocol should be prepended to the `Dataset` path
    argument(s), and any special backend parameters should be passed
    in a `storage_options` dictionary::

        # Initialize Dataset with s3 parquet data
        dataset = Dataset(
            "s3://bucket/path",
            engine="parquet",
            storage_options={'anon': True, 'use_ssl': False},
        )

    By default, both parquet and csv-based data will be converted to
    a Dask-DataFrame collection with a maximum partition size of
    roughly 12.5 percent of the total memory on a single device.  The
    partition size can be changed to a different fraction of total
    memory on a single device with the `part_mem_fraction` argument.
    Alternatively, a specific byte size can be specified with the
    `part_size` argument::

        # Dataset partitions will be ~10% single-GPU memory (or smaller)
        dataset = Dataset("bigfile.parquet", part_mem_fraction=0.1)

        # Dataset partitions will be ~1GB (or smaller)
        dataset = Dataset("bigfile.parquet", part_size="1GB")

    Note that, if both the fractional and literal options are used
    at the same time, `part_size` will take precedence.  Also, for
    parquet-formatted data, the partitioning is done at the row-
    group level, and the byte-size of the first row-group (after
    CuDF conversion) is used to map all other partitions.
    Therefore, if the distribution of row-group sizes is not
    uniform, the partition sizes will not be balanced.

    In addition to handling data stored on disk, a `Dataset` object
    can also be initialized from an existing CuDF/Pandas DataFrame,
    or from a Dask-DataFrame collection (e.g. `dask_cudf.DataFrame`).
    For these in-memory formats, the size/number of partitions will
    not be modified.  That is, a CuDF/Pandas DataFrame (or PyArrow
    Table) will produce a single-partition collection, while the
    number/size of a Dask-DataFrame collection will be preserved::

        # Initialize from CuDF DataFrame (creates 1 partition)
        gdf = cudf.DataFrame(...)
        dataset = Dataset(gdf)

        # Initialize from Dask-CuDF DataFrame (preserves partitions)
        ddf = dask_cudf.read_parquet(...)
        dataset = Dataset(ddf)

    Since the `Dataset` API can both ingest and output a Dask
    collection, it is straightforward to transform data either before
    or after an NVTabular workflow is executed. This means that some
    complex pre-processing operations, that are not yet supported
    in NVTabular, can still be accomplished with the Dask-CuDF API::

        # Sort input data before final Dataset initialization
        # Warning: Global sorting requires significant device memory!
        ddf = Dataset("/path/to/data_pq", engine="parquet").to_ddf()
        ddf = ddf.sort_values("user_rank", ignore_index=True)
        dataset = Dataset(ddf)

    `Dataset Optimization Tips (DOTs)`

    The NVTabular dataset should be created from Parquet files in order
    to get the best possible performance, preferably with a row group size
    of around 128MB.  While NVTabular also supports reading from CSV files,
    reading CSV can be over twice as slow as reading from Parquet. Take a
    look at this notebook_ for an example of transforming the original Criteo
    CSV dataset into a new Parquet dataset optimized for use with NVTabular.

    .. _notebook: https://github.com/NVIDIA/NVTabular/blob/main/examples/optimize_criteo.ipynb


    Parameters
    -----------
    path_or_source : str, list of str, or <dask.dataframe|cudf|pd>.DataFrame
        Dataset path (or list of paths), or a DataFrame. If string,
        should specify a specific file or directory path. If this is a
        directory path, the directory structure must be flat (nested
        directories are not yet supported).
    engine : str or DatasetEngine
        DatasetEngine object or string identifier of engine. Current
        string options include: ("parquet", "csv", "avro"). This argument
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
        is ignored if path_or_source is a DataFrame type. If
        ``cpu=True``, this value will be relative to the total
        host memory detected by the client process.
    storage_options: None or dict
        Further parameters to pass to the bytes backend. This argument
        is ignored if path_or_source is a DataFrame type.
    cpu : bool
        WARNING: Experimental Feature!
        Whether NVTabular should keep all data in cpu memory when
        the Dataset is converted to an internal Dask collection. The
        default value is False, unless ``cudf`` and ``dask_cudf``
        are not installed (in which case the default is True). In the
        future, if True, NVTabular will NOT use any available GPU
        devices for down-stream processing.
        NOTE: Down-stream ops and output do not yet support a
        Dataset generated with ``cpu=True``.
    """

    def __init__(
        self,
        path_or_source,
        engine=None,
        part_size=None,
        part_mem_fraction=None,
        storage_options=None,
        dtypes=None,
        client=None,
        cpu=None,
        **kwargs,
    ):
        self.dtypes = dtypes
        self.client = client

        # Check if we are keeping data in cpu memory
        self.cpu = cpu or False

        # For now, lets warn the user that "cpu mode" is experimental
        if self.cpu:
            warnings.warn(
                "Initializing an NVTabular Dataset in CPU mode."
                "This is an experimental feature with extremely limited support!"
            )

        if isinstance(path_or_source, (dask.dataframe.DataFrame, cudf.DataFrame, pd.DataFrame)):
            # User is passing in a <dask.dataframe|cudf|pd>.DataFrame
            # Use DataFrameDatasetEngine
            moved_collection = (
                False  # Whether a pd-backed collection was moved to cudf (or vice versa)
            )
            if self.cpu:
                if isinstance(path_or_source, pd.DataFrame):
                    # Convert pandas DataFrame to pandas-backed dask.dataframe.DataFrame
                    path_or_source = dask.dataframe.from_pandas(path_or_source, npartitions=1)
                elif isinstance(path_or_source, cudf.DataFrame):
                    # Convert cudf DataFrame to pandas-backed dask.dataframe.DataFrame
                    path_or_source = dask.dataframe.from_pandas(
                        path_or_source.to_pandas(), npartitions=1
                    )
                elif isinstance(path_or_source, dask_cudf.DataFrame):
                    # Convert dask_cudf DataFrame to pandas-backed dask.dataframe.DataFrame
                    path_or_source = path_or_source.to_dask_dataframe()
                    moved_collection = True
            else:
                if isinstance(path_or_source, cudf.DataFrame):
                    # Convert cudf DataFrame to dask_cudf.DataFrame
                    path_or_source = dask_cudf.from_cudf(path_or_source, npartitions=1)
                elif isinstance(path_or_source, pd.DataFrame):
                    # Convert pandas DataFrame to dask_cudf.DataFrame
                    path_or_source = dask_cudf.from_cudf(
                        cudf.from_pandas(path_or_source), npartitions=1
                    )
                elif not isinstance(path_or_source, dask_cudf.DataFrame):
                    # Convert dask.dataframe.DataFrame DataFrame to dask_cudf.DataFrame
                    path_or_source = dask_cudf.from_dask_dataframe(path_or_source)
                    moved_collection = True
            if part_size:
                warnings.warn("part_size is ignored for DataFrame input.")
            if part_mem_fraction:
                warnings.warn("part_mem_fraction is ignored for DataFrame input.")
            self.engine = DataFrameDatasetEngine(
                path_or_source, cpu=self.cpu, moved_collection=moved_collection
            )
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
                        paths, part_size, storage_options=storage_options, cpu=self.cpu, **kwargs
                    )
                elif engine == "csv":
                    self.engine = CSVDatasetEngine(
                        paths, part_size, storage_options=storage_options, cpu=self.cpu, **kwargs
                    )
                elif engine == "avro":
                    try:
                        from .avro import AvroDatasetEngine
                    except ImportError:
                        raise RuntimeError(
                            "Failed to import AvroDatasetEngine. Make sure uavro is installed."
                        )

                    self.engine = AvroDatasetEngine(
                        paths, part_size, storage_options=storage_options, cpu=self.cpu, **kwargs
                    )
                else:
                    raise ValueError("Only parquet, csv, and avro supported (for now).")
            else:
                self.engine = engine(
                    paths, part_size, cpu=self.cpu, storage_options=storage_options
                )

    def to_ddf(self, columns=None, shuffle=False, seed=None):
        """Convert `Dataset` object to `dask_cudf.DataFrame`

        Parameters
        -----------
        columns : str or list(str); default None
            Columns to include in output `DataFrame`. If not specified,
            the output will contain all known columns in the Dataset.
        shuffle : bool; default False
            Whether to shuffle the order of partitions in the output
            `dask_cudf.DataFrame`.  Note that this does not shuffle
            the rows within each partition. This is because the data
            is not actually loaded into memory for this operation.
        seed : int; Optional
            The random seed to use if `shuffle=True`.  If nothing
            is specified, the current system time will be used by the
            `random` std library.
        """
        # Use DatasetEngine to create ddf
        ddf = self.engine.to_ddf(columns=columns)

        # Shuffle the partitions of ddf (optional)
        if shuffle and ddf.npartitions > 1:
            # Start with ordered partitions
            inds = list(range(ddf.npartitions))

            # Use random std library to reorder partitions
            random.seed(seed)
            random.shuffle(inds)

            # Construct new high-level graph (HLG)
            name = ddf._name
            new_name = "shuffle-partitions-" + tokenize(ddf)
            dsk = {(new_name, i): (lambda x: x, (name, ind)) for i, ind in enumerate(inds)}

            new_graph = HighLevelGraph.from_collections(new_name, dsk, dependencies=[ddf])

            # Convert the HLG to a Dask collection
            divisions = [None] * (ddf.npartitions + 1)
            ddf = new_dd_object(new_graph, new_name, ddf._meta, divisions)

        # Special dtype conversion (optional)
        if self.dtypes:
            _meta = _set_dtypes(ddf._meta, self.dtypes)
            return ddf.map_partitions(_set_dtypes, self.dtypes, meta=_meta)
        return ddf

    def to_cpu(self):
        warnings.warn(
            "Changing an NVTabular Dataset to CPU mode."
            "This is an experimental feature with extremely limited support!"
        )
        self.cpu = True
        self.engine.to_cpu()

    def to_gpu(self):
        self.cpu = False
        self.engine.to_gpu()

    def to_iter(self, columns=None, indices=None, shuffle=False, seed=None):
        """Convert `Dataset` object to a `cudf.DataFrame` iterator.

        Note that this method will use `to_ddf` to produce a
        `dask_cudf.DataFrame`, and materialize a single partition for
        each iteration.

        Parameters
        ----------
        columns : str or list(str); default None
            Columns to include in each `DataFrame`. If not specified,
            the outputs will contain all known columns in the Dataset.
        indices : list(int); default None
            A specific list of partition indices to iterate over. If
            nothing is specified, all partitions will be returned in
            order (or the shuffled order, if `shuffle=True`).
        shuffle : bool; default False
            Whether to shuffle the order of `dask_cudf.DataFrame`
            partitions used by the iterator.  If the `indices`
            argument is specified, those indices correspond to the
            partition indices AFTER the shuffle operation.
        seed : int; Optional
            The random seed to use if `shuffle=True`.  If nothing
            is specified, the current system time will be used by the
            `random` std library.
        """
        if isinstance(columns, str):
            columns = [columns]

        return DataFrameIter(
            self.to_ddf(columns=columns, shuffle=shuffle, seed=seed), indices=indices
        )

    def to_parquet(
        self,
        output_path,
        shuffle=None,
        out_files_per_proc=None,
        num_threads=0,
        dtypes=None,
        cats=None,
        conts=None,
        labels=None,
    ):
        """Writes out to a parquet dataset

        Parameters
        ----------
        output_path : string
            Path to write processed/shuffled output data
        shuffle : nvt.io.Shuffle enum
            How to shuffle the output dataset. Shuffling is only
            performed if the data is written to disk. For all options,
            other than `None` (which means no shuffling), the partitions
            of the underlying dataset/ddf will be randomly ordered. If
            `PER_PARTITION` is specified, each worker/process will also
            shuffle the rows within each partition before splitting and
            appending the data to a number (`out_files_per_proc`) of output
            files. Output files are distinctly mapped to each worker process.
            If `PER_WORKER` is specified, each worker will follow the same
            procedure as `PER_PARTITION`, but will re-shuffle each file after
            all data is persisted.  This results in a full shuffle of the
            data processed by each worker.  To improve performace, this option
            currently uses host-memory `BytesIO` objects for the intermediate
            persist stage. The `FULL` option is not yet implemented.
        out_files_per_proc : integer
            Number of files to create (per process) after
            shuffling the data
        num_threads : integer
            Number of IO threads to use for writing the output dataset.
            For `0` (default), no dedicated IO threads will be used.
        dtypes : dict
            Dictionary containing desired datatypes for output columns.
            Keys are column names, values are datatypes.
        cats : list of str, optional
            List of categorical columns
        conts : list of str, optional
            List of continuous columns
        labels : list of str, optional
            List of label columns
        """

        shuffle = _check_shuffle_arg(shuffle)
        ddf = self.to_ddf(shuffle=shuffle)

        if dtypes:
            _meta = _set_dtypes(ddf._meta, dtypes)
            ddf = ddf.map_partitions(_set_dtypes, dtypes, meta=_meta)

        fs = get_fs_token_paths(output_path)[0]
        fs.mkdirs(output_path, exist_ok=True)

        # Output dask_cudf DataFrame to dataset
        _ddf_to_dataset(
            ddf,
            fs,
            output_path,
            shuffle,
            out_files_per_proc,
            cats or [],
            conts or [],
            labels or [],
            "parquet",
            self.client,
            num_threads,
            self.cpu,
        )

    def to_hugectr(
        self,
        output_path,
        cats,
        conts,
        labels,
        shuffle=None,
        out_files_per_proc=None,
        num_threads=0,
        dtypes=None,
    ):
        """Writes out to a parquet dataset

        Parameters
        ----------
        output_path : string
            Path to write processed/shuffled output data
        cats : list of str
            List of categorical columns
        conts : list of str
            List of continuous columns
        labels : list of str
            List of label columns
        shuffle : nvt.io.Shuffle, optional
            How to shuffle the output dataset. Shuffling is only
            performed if the data is written to disk. For all options,
            other than `None` (which means no shuffling), the partitions
            of the underlying dataset/ddf will be randomly ordered. If
            `PER_PARTITION` is specified, each worker/process will also
            shuffle the rows within each partition before splitting and
            appending the data to a number (`out_files_per_proc`) of output
            files. Output files are distinctly mapped to each worker process.
            If `PER_WORKER` is specified, each worker will follow the same
            procedure as `PER_PARTITION`, but will re-shuffle each file after
            all data is persisted.  This results in a full shuffle of the
            data processed by each worker.  To improve performace, this option
            currently uses host-memory `BytesIO` objects for the intermediate
            persist stage. The `FULL` option is not yet implemented.
        out_files_per_proc : integer
            Number of files to create (per process) after
            shuffling the data
        num_threads : integer
            Number of IO threads to use for writing the output dataset.
            For `0` (default), no dedicated IO threads will be used.
        dtypes : dict
            Dictionary containing desired datatypes for output columns.
            Keys are column names, values are datatypes.
        """

        # For now, we must move to the GPU to
        # write an output dataset.
        # TODO: Support CPU-mode output
        self.to_gpu()

        shuffle = _check_shuffle_arg(shuffle)
        shuffle = _check_shuffle_arg(shuffle)
        ddf = self.to_ddf(shuffle=shuffle)
        if dtypes:
            _meta = _set_dtypes(ddf._meta, dtypes)
            ddf = ddf.map_partitions(_set_dtypes, dtypes, meta=_meta)

        fs = get_fs_token_paths(output_path)[0]
        fs.mkdirs(output_path, exist_ok=True)

        # Output dask_cudf DataFrame to dataset,
        _ddf_to_dataset(
            ddf,
            fs,
            output_path,
            shuffle,
            out_files_per_proc,
            cats,
            conts,
            labels,
            "hugectr",
            self.client,
            num_threads,
            self.cpu,
        )

    @property
    def num_rows(self):
        return self.engine.num_rows

    def validate_dataset(self, **kwargs):
        """Validate for efficient processing.

        The purpose of this method is to validate that the Dataset object
        meets the minimal requirements for efficient NVTabular processing.
        For now, this criteria requires the data to be in parquet format.

        Example Usage::

            dataset = Dataset("/path/to/data_pq", engine="parquet")
            assert validate_dataset(dataset)

        Parameters
        -----------
        **kwargs :
            Key-word arguments to pass down to the engine's validate_dataset
            method. For the recommended parquet format, these arguments
            include `add_metadata_file`, `row_group_max_size`, `file_min_size`,
            and `require_metadata_file`. For more information, see
            `ParquetDatasetEngine.validate_dataset`.

        Returns
        -------
        valid : bool
            Whether or not the input dataset is valid for efficient NVTabular
            processing.
        """

        # Check that the dataset format is Parquet
        if not isinstance(self.engine, ParquetDatasetEngine):
            msg = (
                "NVTabular is optimized for the parquet format. Please use "
                "the regenerate_dataset method to convert your dataset."
            )
            warnings.warn(msg)
            return False  # Early return

        return self.engine.validate_dataset(**kwargs)

    def regenerate_dataset(
        self, output_path, columns=None, output_format="parquet", compute=True, **kwargs
    ):
        """Regenerate an NVTabular Dataset for efficient processing by writing
        out new Parquet files. (This method preserves the original ordering,
        while ``to_parquet`` does not.)

        Example Usage::

            dataset = Dataset("/path/to/data_pq", engine="parquet")
            dataset.regenerate_dataset(
                out_path, part_size="1MiB", file_size="10MiB"
            )

        Parameters
        -----------
        output_path : string
            Root directory path to use for the new (regenerated) dataset.
        columns : list(string), optional
            Subset of columns to include in the regenerated dataset.
        output_format : string, optional
            Format to use for regenerated dataset.  Only "parquet" (default)
            is currently supported.
        compute : bool, optional
            Whether to compute the task graph or to return a Delayed object.
            By default, the graph will be executed.
        **kwargs :
            Key-word arguments to pass down to the engine's regenerate_dataset
            method. See `ParquetDatasetEngine.regenerate_dataset` for more
            information.

        Returns
        -------
        result : int or Delayed
            If `compute=True` (default), the return value will be an integer
            corresponding to the number of generated data files.  If `False`,
            the returned value will be a `Delayed` object.
        """

        # Check that the desired output format is Parquet
        if output_format not in ["parquet"]:
            msg = (
                f"NVTabular is optimized for the parquet format. "
                f"{output_format} is not yet a supported output format for "
                f"regenerate_dataset."
            )
            raise ValueError(msg)

        result = ParquetDatasetEngine.regenerate_dataset(self, output_path, columns=None, **kwargs)
        if compute:
            return result.compute()
        else:
            return result


def _set_dtypes(chunk, dtypes):
    for col, dtype in dtypes.items():
        if type(dtype) is str:
            if "hex" in dtype and chunk[col].dtype == "object":
                chunk[col] = chunk[col].str.htoi()
                chunk[col] = chunk[col].astype(np.int32)
        else:
            chunk[col] = chunk[col].astype(dtype)
    return chunk


class DataFrameIter:
    def __init__(self, ddf, columns=None, indices=None):
        self.indices = indices if isinstance(indices, list) else range(ddf.npartitions)
        self._ddf = ddf
        self.columns = columns

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for i in self.indices:
            part = self._ddf.get_partition(i)
            if self.columns:
                yield part[self.columns].compute(scheduler="synchronous")
            else:
                yield part.compute(scheduler="synchronous")
            part = None
