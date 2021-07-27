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

import collections
import logging
import math
import random
import warnings

import dask
import numpy as np
from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.highlevelgraph import HighLevelGraph
from dask.utils import natural_sort_key, parse_bytes
from fsspec.core import get_fs_token_paths
from fsspec.utils import stringify_path

from nvtabular.dispatch import _convert_data, _hex_to_int, _is_dataframe_object
from nvtabular.io.shuffle import _check_shuffle_arg
from nvtabular.utils import global_dask_client

from ..utils import device_mem_size
from .csv import CSVDatasetEngine
from .dask import _ddf_to_dataset, _simple_shuffle
from .dataframe_engine import DataFrameDatasetEngine
from .parquet import ParquetDatasetEngine

try:
    import cudf
except ImportError:
    cudf = None

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
    npartitions : int
        Desired number of Dask-collection partitions to produce in
        the ``to_ddf`` method when ``path_or_source`` corresponds to a
        DataFrame type.  This argument is ignored for file-based
        ``path_or_source`` input.
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
    base_dataset : Dataset
        Optional reference to the original "base" Dataset object used
        to construct the current Dataset instance.  This object is
        used to preserve file-partition mapping information.
    """

    def __init__(
        self,
        path_or_source,
        engine=None,
        npartitions=None,
        part_size=None,
        part_mem_fraction=None,
        storage_options=None,
        dtypes=None,
        client=None,
        cpu=None,
        base_dataset=None,
        **kwargs,
    ):
        self.dtypes = dtypes
        self.client = client

        # Check if we are keeping data in cpu memory
        self.cpu = cpu
        if not self.cpu:
            self.cpu = cudf is None

        # Keep track of base dataset (optional)
        self.base_dataset = base_dataset or self

        # For now, lets warn the user that "cpu mode" is experimental
        if self.cpu:
            warnings.warn(
                "Initializing an NVTabular Dataset in CPU mode."
                "This is an experimental feature with extremely limited support!"
            )

        npartitions = npartitions or 1
        if isinstance(path_or_source, dask.dataframe.DataFrame) or _is_dataframe_object(
            path_or_source
        ):
            # User is passing in a <dask.dataframe|cudf|pd>.DataFrame
            # Use DataFrameDatasetEngine
            _path_or_source = _convert_data(
                path_or_source, cpu=self.cpu, to_collection=True, npartitions=npartitions
            )
            # Check if this is a collection that has now moved between host <-> device
            moved_collection = isinstance(path_or_source, dask.dataframe.DataFrame) and (
                not isinstance(_path_or_source._meta, type(path_or_source._meta))
            )
            if part_size:
                warnings.warn("part_size is ignored for DataFrame input.")
            if part_mem_fraction:
                warnings.warn("part_mem_fraction is ignored for DataFrame input.")
            self.engine = DataFrameDatasetEngine(
                _path_or_source, cpu=self.cpu, moved_collection=moved_collection
            )
        else:
            if part_size:
                # If a specific partition size is given, use it directly
                part_size = parse_bytes(part_size)
            else:
                # If a fractional partition size is given, calculate part_size
                part_mem_fraction = part_mem_fraction or 0.125
                assert 0.0 < part_mem_fraction < 1.0
                if part_mem_fraction > 0.25:
                    warnings.warn(
                        "Using very large partitions sizes for Dask. "
                        "Memory-related errors are likely."
                    )
                part_size = int(device_mem_size(kind="total", cpu=self.cpu) * part_mem_fraction)

            # Engine-agnostic path handling
            paths = path_or_source
            if hasattr(paths, "name"):
                paths = stringify_path(paths)
            if isinstance(paths, str):
                paths = [paths]
            paths = sorted(paths, key=natural_sort_key)

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
                    except ImportError as e:
                        raise RuntimeError(
                            "Failed to import AvroDatasetEngine. Make sure uavro is installed."
                        ) from e

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

    @property
    def file_partition_map(self):
        return self.engine._file_partition_map

    @property
    def partition_lens(self):
        return self.engine._partition_lens

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

    def shuffle_by_keys(self, keys, hive_data=None, npartitions=None):
        """Shuffle the in-memory Dataset so that all unique-key
        combinations are moved to the same partition.

        Parameters
        ----------
        keys : list(str)
            Column names to shuffle by.
        hive_data : bool; default None
            Whether the dataset is backed by a hive-partitioned
            dataset (with the keys encoded in the directory structure).
            By default, the Dataset's `file_partition_map` property will
            be inspected to infer this setting. When `hive_data` is True,
            the number of output partitions will correspond to the number
            of unique key combinations in the dataset.
        npartitions : int; default None
            Number of partitions in the output Dataset. For hive-partitioned
            data, this value should be <= the number of unique key
            combinations (the default), otherwise it will be ignored. For
            data that is not hive-partitioned, the ``npartitions`` input
            should be <= the orginal partition count, otherwise it will be
            ignored.
        """

        # Make sure we are dealing with a list
        keys = [keys] if not isinstance(keys, (list, tuple)) else keys

        # Start with default ddf
        ddf = self.to_ddf()
        if npartitions:
            npartitions = min(ddf.npartitions, npartitions)

        if hive_data is not False:
            # The keys may be encoded in the directory names.
            # Let's use the file_partition_map to extract this info.
            try:
                _mapping = self.file_partition_map
            except AttributeError as e:
                _mapping = None
                if hive_data:
                    raise RuntimeError("Failed to extract hive-partition mapping!") from e

            # If we have a `_mapping` available, check if the
            # file names include information about all our keys
            hive_mapping = collections.defaultdict(list)
            if _mapping:
                for k, v in _mapping.items():
                    for part in k.split(self.engine.fs.sep)[:-1]:
                        try:
                            _key, _val = part.split("=")
                        except ValueError:
                            continue
                        if _key in keys:
                            hive_mapping[_key].append(_val)

            if set(hive_mapping.keys()) == set(keys):

                # Generate hive-mapping DataFrame summary
                hive_mapping = type(ddf._meta)(hive_mapping)
                cols = list(hive_mapping.columns)
                for c in keys:
                    typ = ddf._meta[c].dtype
                    if c in cols:
                        hive_mapping[c] = hive_mapping[c].astype(typ)

                # Generate simple-shuffle plan
                target_mapping = hive_mapping.drop_duplicates().reset_index(drop=True)
                target_mapping.index.name = "_partition"
                hive_mapping.index.name = "_sort"
                target_mapping.reset_index(drop=False, inplace=True)
                plan = (
                    hive_mapping.reset_index()
                    .merge(target_mapping, on=cols, how="left")
                    .sort_values("_sort")["_partition"]
                )

                if hasattr(plan, "to_pandas"):
                    plan = plan.to_pandas()

                # Deal with repartitioning
                if npartitions and npartitions < len(target_mapping):
                    q = np.linspace(0.0, 1.0, num=npartitions + 1)
                    divs = plan.quantile(q)
                    partitions = divs.searchsorted(plan, side="right") - 1
                    partitions[(plan >= divs.iloc[-1]).values] = len(divs) - 2
                    plan = partitions.tolist()
                elif len(plan) != len(plan.unique()):
                    plan = plan.to_list()
                else:
                    # Plan is a unique 1:1 ddf partition mapping.
                    # We already have shuffled data.
                    return self

                # TODO: We should avoid shuffling the original ddf and
                # instead construct a new (more-efficent) graph to read
                # multiple files from each partition directory at once.
                # Generally speaking, we can optimize this code path
                # much further.
                return Dataset(_simple_shuffle(ddf, plan), client=self.client)

        # Warn user if there is an unused global
        # Dask client available
        if global_dask_client(self.client):
            warnings.warn(
                "A global dask.distributed client has been detected, but the "
                "single-threaded scheduler is being used for this shuffle operation. "
                "Please use the `client` argument to initialize a `Dataset` and/or "
                "`Workflow` object with distributed-execution enabled."
            )

        # Fall back to dask.dataframe algorithm
        return Dataset(ddf.shuffle(keys, npartitions=npartitions), client=self.client)

    def repartition(self, npartitions=None, partition_size=None):
        """Repartition the underlying ddf, and return a new Dataset

        Parameters
        ----------
        npartitions : int; default None
            Number of partitions in output ``Dataset``. Only used if
            ``partition_size`` isn’t specified.
        partition_size : int or str; default None
            Max number of bytes of memory for each partition. Use
            numbers or strings like '5MB'. If specified, ``npartitions``
            will be ignored.
        """
        return Dataset(
            self.to_ddf()
            .clear_divisions()
            .repartition(
                npartitions=npartitions,
                partition_size=partition_size,
            )
        )

    @classmethod
    def merge(cls, left, right, **kwargs):
        """Merge two Dataset objects

        Produces a new Dataset object. If the ``cpu`` Dataset attributes
        do not match, the right side will be modified. See Dask-Dataframe
        ``merge`` documentation for more information. Example usage::

            ds_1 = Dataset("file.parquet")
            ds_2 = Dataset(cudf.DataFrame(...))
            ds_merged = Dataset.merge(ds_1, ds_2, on="foo", how="inner")

        Parameters
        ----------
        left : Dataset
            Left-side Dataset object.
        right : Dataset
            Right-side Dataset object.
        **kwargs :
            Key-word arguments to be passed through to Dask-Dataframe.
        """

        # Ensure both Dataset objects are eith cudf or pandas based
        if left.cpu and not right.cpu:
            _right = cls(right.to_ddf())
            _right.to_cpu()
        elif not left.cpu and right.cpu:
            _right = cls(right.to_ddf())
            _right.to_gpu()

        return cls(
            left.to_ddf()
            .clear_divisions()
            .merge(
                _right.to_ddf().clear_divisions(),
                **kwargs,
            )
        )

    def to_iter(self, columns=None, indices=None, shuffle=False, seed=None, use_file_metadata=None):
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
        use_file_metadata : bool; Optional
            Whether to allow the returned ``DataFrameIter`` object to
            use file metadata from the ``base_dataset`` to estimate
            the row-count. By default, the file-metadata
            optimization will only be used if the current Dataset is
            backed by a file-based engine. Otherwise, it is possible
            that an intermediate transform has modified the row-count.
        """
        if isinstance(columns, str):
            columns = [columns]

        # Try to extract the row-size metadata
        # if we are not shuffling
        partition_lens_meta = None
        if not shuffle and use_file_metadata is not False:
            # We are allowed to use file metadata to calculate
            # partition sizes.  If `use_file_metadata` is None,
            # we only use metadata if `self` is backed by a
            # file-based engine (like "parquet").  Otherwise,
            # we cannot be "sure" that the metadata row-count
            # is correct.
            try:
                if use_file_metadata:
                    partition_lens_meta = self.base_dataset.partition_lens
                else:
                    partition_lens_meta = self.partition_lens
            except AttributeError:
                pass

        return DataFrameIter(
            self.to_ddf(columns=columns, shuffle=shuffle, seed=seed),
            indices=indices,
            partition_lens=partition_lens_meta,
        )

    def to_parquet(
        self,
        output_path,
        shuffle=None,
        preserve_files=False,
        output_files=None,
        out_files_per_proc=None,
        num_threads=0,
        dtypes=None,
        cats=None,
        conts=None,
        labels=None,
        suffix=".parquet",
        partition_on=None,
        method="subgraph",
    ):
        """Writes out to a parquet dataset

        Parameters
        ----------
        output_path : string
            Path to write processed/shuffled output data
        shuffle : nvt.io.Shuffle enum
            How to shuffle the output dataset. For all options,
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
        partition_on : str or list(str)
            Columns to use for hive-partitioning.  If this option is used,
            `preserve_files`, `output_files`, and `out_files_per_proc` will
            all be ignored.  Also, the `PER_WORKER` shuffle will not be
            supported.
        preserve_files : bool
            Whether to preserve the original file-to-partition mapping of
            the base dataset. This option requires `method="subgraph"`, and is
            only available if the base dataset is known, and if it corresponds
            to csv or parquet format. If True, the `out_files_per_proc` option
            will be ignored. Default is False.
        output_files : dict, list or int
            The total number of desired output files. This option requires
            `method="subgraph"`, and the default value will be the number of Dask
            workers, multiplied by `out_files_per_proc`. For further output-file
            control, this argument may also be used to pass a dictionary mapping
            the output file names to partition indices, or a list of desired
            output-file names.
        out_files_per_proc : integer
            Number of output files that each process will use to shuffle an input
            partition. Deafult is 1. If `method="worker"`, the total number of output
            files will always be the total number of Dask workers, multiplied by this
            argument. If `method="subgraph"`, the total number of files is determined
            by `output_files` (and `out_files_per_proc` must be 1 if a dictionary is
            specified).
        num_threads : integer
            Number of IO threads to use for writing the output dataset.
            For `0` (default), no dedicated IO threads will be used.
        dtypes : dict
            Dictionary containing desired datatypes for output columns.
            Keys are column names, values are datatypes.
        suffix : str or False
            File-name extension to use for all output files. This argument
            is ignored if a specific list of file names is specified using
            the ``output_files`` option. If ``preserve_files=True``, this
            suffix will be appended to the original name of each file,
            unless the original extension is ".csv", ".parquet", ".avro",
            or ".orc" (in which case the old extension will be replaced).
        cats : list of str, optional
            List of categorical columns
        conts : list of str, optional
            List of continuous columns
        labels : list of str, optional
            List of label columns
        method : {"subgraph", "worker"}
            General algorithm to use for the parallel graph execution. In order
            to minimize memory pressure, `to_parquet` will use a `"subgraph"` by
            default. This means that we segment the full Dask task graph into a
            distinct subgraph for each output file (or output-file group). Then,
            each of these subgraphs is executed, in full, by the same worker (as
            a single large task). In some cases, it may be more ideal to prioritize
            concurrency. In that case, a worker-based approach can be used by
            specifying `method="worker"`.
        """

        # Check that method (algorithm) is valid
        if method not in ("subgraph", "worker"):
            raise ValueError(f"{method} not a recognized method for `Dataset.to_parquet`")

        # Deal with method-specific defaults
        if method == "worker":
            if output_files or preserve_files:
                raise ValueError("output_files and preserve_files require `method='subgraph'`")
            output_files = False
        elif preserve_files and output_files:
            raise ValueError("Cannot specify both preserve_files and output_files.")
        elif not (output_files or preserve_files):
            # Default "subgraph" behavior - Set output_files to the
            # total umber of workers, multiplied by out_files_per_proc
            try:
                nworkers = len(self.client.cluster.workers)
            except AttributeError:
                nworkers = 1
            output_files = nworkers * (out_files_per_proc or 1)

        # Check shuffle argument
        shuffle = _check_shuffle_arg(shuffle)

        if isinstance(output_files, dict) or (not output_files and preserve_files):
            # Do not shuffle partitions if we are preserving files or
            # if a specific file-partition mapping is already specified
            ddf = self.to_ddf()
        else:
            ddf = self.to_ddf(shuffle=shuffle)

        # Replace None/False suffix argument with ""
        suffix = suffix or ""

        # Deal with `method=="subgraph"`.
        # Convert `output_files` argument to a dict mapping
        if output_files:

            #   NOTES on `output_files`:
            #
            # - If a list of file names is specified, a contiguous range of
            #   output partitions will be mapped to each file. The same
            #   procedure is used if an integer is specified, but the file
            #   names will be written as "part_*".
            #
            # - When `output_files` is used, the `output_files_per_proc`
            #   argument will be interpreted as the desired number of output
            #   files to write within the same task at run time (enabling
            #   input partitions to be shuffled into multiple output files).
            #
            # - Passing a list or integer to `output_files` will preserve
            #   the original ordering of the input data as long as
            #   `out_files_per_proc` is set to `1` (or `None`), and
            #   `shuffle==None`.
            #
            # - If a dictionary is specified, excluded partition indices
            #   will not be written to disk.
            #
            # - To map multiple output files to a range of input partitions,
            #   dictionary-input keys should correspond to a tuple of file
            #   names.

            # Use out_files_per_proc to calculate how
            # many output files should be written within the
            # same subgraph.  Note that we must a
            files_per_task = out_files_per_proc or 1
            required_npartitions = ddf.npartitions
            if isinstance(output_files, int):
                required_npartitions = output_files
                files_per_task = min(files_per_task, output_files)
            elif isinstance(output_files, list):
                required_npartitions = len(output_files)
                files_per_task = min(files_per_task, len(output_files))
            elif out_files_per_proc:
                raise ValueError(
                    "Cannot specify out_files_per_proc if output_files is "
                    "defined as a dictionary mapping. Please define each "
                    "key in output_files as a tuple of file names if you "
                    "wish to have those files written by the same process."
                )

            # Repartition ddf if necessary
            if ddf.npartitions < required_npartitions:
                ddf = ddf.clear_divisions().repartition(npartitions=required_npartitions)

            # Construct an output_files dictionary if necessary
            if isinstance(output_files, int):
                output_files = [f"part_{i}" + suffix for i in range(output_files)]
            if isinstance(output_files, list):
                new = {}
                split = math.ceil(ddf.npartitions / len(output_files))
                for i in range(0, len(output_files), files_per_task):
                    fns = output_files[i : i + files_per_task]
                    start = i * split
                    stop = min(start + split * len(fns), ddf.npartitions)
                    new[tuple(fns)] = np.arange(start, stop)
                output_files = new
                suffix = ""  # Don't add a suffix later - Names already include it
            if not isinstance(output_files, dict):
                raise TypeError(f"{type(output_files)} not a supported type for `output_files`.")

        # If we are preserving files, use the stored dictionary,
        # or use file_partition_map to extract the mapping
        elif preserve_files:
            try:
                _output_files = self.base_dataset.file_partition_map
            except AttributeError as e:
                raise AttributeError(
                    f"`to_parquet(..., preserve_files=True)` is not currently supported "
                    f"for datasets with a {type(self.base_dataset.engine)} engine. Check "
                    f"that `dataset.base_dataset` is backed by csv or parquet files."
                ) from e
            if suffix == "":
                output_files = _output_files
            else:
                output_files = {}
                for fn, rgs in _output_files.items():
                    split_fn = fn.split(".")
                    if split_fn[-1] in ("parquet", "avro", "orc", "csv"):
                        output_files[".".join(split_fn[:-1]) + suffix] = rgs
                    else:
                        output_files[fn + suffix] = rgs
            suffix = ""  # Don't add a suffix later - Names already include it

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
            output_files,
            out_files_per_proc,
            cats or [],
            conts or [],
            labels or [],
            "parquet",
            self.client,
            num_threads,
            self.cpu,
            suffix=suffix,
            partition_on=partition_on,
        )

    def to_hugectr(
        self,
        output_path,
        cats,
        conts,
        labels,
        shuffle=None,
        file_partition_map=None,
        out_files_per_proc=None,
        num_threads=0,
        dtypes=None,
    ):
        """Writes out to a hugectr dataset

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
        file_partition_map : dict
            Dictionary mapping of output file names to partition indices
            that should be written to that file name.  If this argument
            is passed, only the partitions included in the dictionary
            will be written to disk, and the `output_files_per_proc` argument
            will be ignored.
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
            file_partition_map,
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

    @property
    def npartitions(self):
        return self.to_ddf().npartitions

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
                "the to_parquet method to convert your dataset."
            )
            warnings.warn(msg)
            return False  # Early return

        return self.engine.validate_dataset(**kwargs)

    def regenerate_dataset(
        self,
        output_path,
        columns=None,
        output_format="parquet",
        compute=True,
        **kwargs,
    ):
        """EXPERIMENTAL:
        Regenerate an NVTabular Dataset for efficient processing by writing
        out new Parquet files. In contrast to default ``to_parquet`` behavior,
        this method preserves the original ordering.

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

    @classmethod
    def _bind_dd_method(cls, name):
        """Bind Dask-Dataframe method to the Dataset class"""

        def meth(self, *args, **kwargs):
            _meth = getattr(self.to_ddf(), name)
            return _meth(*args, **kwargs)

        meth.__name__ = name
        setattr(cls, name, meth)


# Bind (simple) Dask-Dataframe Methods
for op in ["compute", "persist", "head", "tail"]:
    Dataset._bind_dd_method(op)


def _set_dtypes(chunk, dtypes):
    for col, dtype in dtypes.items():
        if isinstance(dtype, str) and ("hex" in dtype):
            chunk[col] = _hex_to_int(chunk[col])
        else:
            chunk[col] = chunk[col].astype(dtype)
    return chunk


class DataFrameIter:
    def __init__(self, ddf, columns=None, indices=None, partition_lens=None):
        self.indices = indices if isinstance(indices, list) else range(ddf.npartitions)
        self._ddf = ddf
        self.columns = columns
        self.partition_lens = partition_lens

    def __len__(self):
        if self.partition_lens:
            # Use metadata-based partition-size information
            # if/when it is available.  Note that this metadata
            # will not be correct if rows where added or dropped
            # after IO (within Ops).
            return sum(self.partition_lens[i] for i in self.indices)
        if len(self.indices) < self._ddf.npartitions:
            return len(self._ddf.partitions[self.indices])
        return len(self._ddf)

    def __iter__(self):
        for i in self.indices:
            part = self._ddf.get_partition(i)
            if self.columns:
                yield part[self.columns].compute(scheduler="synchronous")
            else:
                yield part.compute(scheduler="synchronous")
        part = None
