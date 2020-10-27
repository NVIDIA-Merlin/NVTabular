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
from fsspec.utils import stringify_path

from ..utils import device_mem_size
from .csv import CSVDatasetEngine
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

    Dataset Optimization Tips (DOTs)
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
        is ignored if path_or_source is a DataFrame type.
    storage_options: None or dict
        Further parameters to pass to the bytes backend. This argument
        is ignored if path_or_source is a DataFrame type.
    """

    def __init__(
        self,
        path_or_source,
        engine=None,
        part_size=None,
        part_mem_fraction=None,
        storage_options=None,
        dtypes=None,
        **kwargs,
    ):
        self.dtypes = dtypes
        if isinstance(path_or_source, (dask.dataframe.DataFrame, cudf.DataFrame, pd.DataFrame)):
            # User is passing in a <dask.dataframe|cudf|pd>.DataFrame
            # Use DataFrameDatasetEngine
            if isinstance(path_or_source, cudf.DataFrame):
                path_or_source = dask_cudf.from_cudf(path_or_source, npartitions=1)
            elif isinstance(path_or_source, pd.DataFrame):
                path_or_source = dask_cudf.from_cudf(
                    cudf.from_pandas(path_or_source), npartitions=1
                )
            elif not isinstance(path_or_source, dask_cudf.DataFrame):
                path_or_source = dask_cudf.from_dask_dataframe(path_or_source)
            if part_size:
                warnings.warn("part_size is ignored for DataFrame input.")
            if part_mem_fraction:
                warnings.warn("part_mem_fraction is ignored for DataFrame input.")
            self.engine = DataFrameDatasetEngine(path_or_source)
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
                        paths, part_size, storage_options=storage_options, **kwargs
                    )
                elif engine == "csv":
                    self.engine = CSVDatasetEngine(
                        paths, part_size, storage_options=storage_options, **kwargs
                    )
                elif engine == "avro":
                    try:
                        from .avro import AvroDatasetEngine
                    except ImportError:
                        raise RuntimeError(
                            "Failed to import AvroDatasetEngine. Make sure uavro is installed."
                        )

                    self.engine = AvroDatasetEngine(
                        paths, part_size, storage_options=storage_options, **kwargs
                    )
                else:
                    raise ValueError("Only parquet and csv supported (for now).")
            else:
                self.engine = engine(paths, part_size, storage_options=storage_options)

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

    def to_iter(self, columns=None, indices=None, shuffle=False, seed=None):
        """Convert `Dataset` object to a `cudf.DataFrame` iterator.

        Note that this method will use `to_ddf` to produce a
        `dask_cudf.DataFrame`, and materialize a single partition for
        each iteration.

        Parameters
        -----------
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

    @property
    def num_rows(self):
        return self.engine.num_rows


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
