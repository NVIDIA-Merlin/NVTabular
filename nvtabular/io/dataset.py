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
    """Dask-based Dataset Class
        Converts a dataset into a dask_cudf DataFrame on demand

    Parameters
    -----------
    path_or_source : str, list of str, or <dask.dataframe|cudf|pd>.DataFrame
        Dataset path (or list of paths), or a DataFrame. If string,
        should specify a specific file or directory path. If this is a
        directory path, the directory structure must be flat (nested
        directories are not yet supported).
    engine : str or DatasetEngine
        DatasetEngine object or string identifier of engine. Current
        string options include: ("parquet", "csv"). This argument
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
