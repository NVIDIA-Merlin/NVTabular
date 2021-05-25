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

import cudf
import cupy
import pandas as pd
import pyarrow as pa
import numpy as np

import dask.dataframe as dd
import dask_cudf

from nvtabular.worker import fetch_table_data, get_worker_cache
from nvtabular.dispatch import DataFrameType, _arange
from nvtabular.io import Dataset

from .operator import ColumnNames, Operator


class JoinExternal(Operator):
    """
    Join each dataset partition to an external table. For performance
    reasons, only "left" and "inner" join transformations are supported.

    Example usage::

        # Load dataset which should be joined to the main dataset
        df_external = cudf.read_parquet('external.parquet')

        # Use JoinExternal to define a NVTabular workflow
        joined = nvt.ColumnGroup(columns_left) >> nvt.ops.JoinExternal(
            df_ext,
            on=['key1', 'key2'],
            on_ext=['key1_ext', 'key2_ext'],
            how='left',
            columns_ext=['key1_ext', 'key2_ext', 'cat1', 'cat2', 'num1'],
            kind_ext='cudf',
            cache='device'
        ) >> ...
        processor = nvtabular.Workflow(joined)

    Parameters
    -----------
    df_ext : DataFrame, pyarrow.Table, Dataset, dd.DataFrame, or file path(s)
        The external table to join to each partition of the dataset. Note
        that the join must be a partition-wise transformation. Therefore,
        if ``df_ext`` is a multi-partition Dask collection, it will need to
        be broadcasted to every partition.
    on : str or list(str)
        Column name(s) to merge on
    how : {"left", "inner"}; default "left"
        Type of join operation to perform.
    on_ext : str or list(str); Optional
        Column name(s) on external table to join on. By default,
        we assume ``on_ext`` is the same as ``on``.
    columns_ext : list(str); Optional
        Subset of columns to select from external table before join.
    drop_duplicates_ext : bool; Default False
        Drop duplicates from external table before join.
    kind_ext : {"arrow", "cudf", "pandas", "parquet", "csv"}
        Format of ``df_ext``.  If nothing is specified, the format
        will be inferred.
    cache : {"device", "host", "disk"}
        Where to cache ``df_ext`` between transformations. Only used
        if the data is originally stored on disk.
    """

    def __init__(
        self,
        df_ext,
        on,
        how="left",
        on_ext=None,
        columns_ext=None,
        drop_duplicates_ext=None,
        kind_ext=None,
        cache="host",
        **kwargs,
    ):
        super(JoinExternal).__init__()
        self.on = on
        self.df_ext = df_ext
        self.on_ext = on_ext or self.on
        self.how = how
        self.kind_ext = kind_ext or _detect_format(self.df_ext)
        self.columns_ext = columns_ext
        self.drop_duplicates_ext = drop_duplicates_ext
        self.cache = cache
        self.kwargs = kwargs
        self.cpu = None
        if self.how not in ("left", "inner"):
            raise ValueError("Only left join is currently supported.")
        if self.kind_ext not in (
            "arrow",
            "cudf",
            "pandas",
            "parquet",
            "csv",
            "dask-dataframe",
            "dask-cudf",
            "dataset",
        ):
            raise ValueError("kind_ext option not recognized.")

    @property
    @functools.lru_cache(1)
    def _ext(self):
        # Define _ext, depending on `kind_ext`
        if self.kind_ext == "dataset":
            # Use Dataset.to_ddf
            _dataset = self.df_ext
            if self.cpu:
                _dataset.to_cpu()
            else:
                _dataset.to_gpu()
            _ext = _check_partition_count(
                _dataset.to_ddf(columns=self.columns_ext)
            )
        elif self.kind_ext in ("cudf", "dask-cudf"):
            if self.cpu:
                # Need to convert GPU DataFrame to CPU Dataframe
                if self.kind_ext == "cudf":
                    _ext = self.df_ext.to_pandas()
                else:
                    _ext = _check_partition_count(
                        self.df_ext.to_dask_dataframe()
                    )
            else:
                # Already in proper GPU format
                _ext = _check_partition_count(self.df_ext)
        elif self.kind_ext in ("pandas", "dask-dataframe"):
            if not self.cpu:
                # Need to convert CPU DataFrame to GPU Dataframe
                if self.kind_ext == "pandas":
                    _ext = cudf.DataFrame.from_pandas(self.df_ext)
                else:
                    _ext = _check_partition_count(
                        self.df_ext.map_partitions(cudf.from_pandas)
                    )
            else:
                # Already in proper CPU format
                _ext = _check_partition_count(self.df_ext)
        elif self.kind_ext == "arrow":
            if self.cpu:
                # Arrow Table to pandas DataFrame
                _ext = self.df_ext.to_pandas()
            else:
                # Arrow Table to cudf DataFrame
                _ext = cudf.DataFrame.from_arrow(self.df_ext)
        else:
            if self.kind_ext == "parquet":
                # Read from parquet dataset
                kwargs = {
                    "split_row_groups": False,
                    "index": False,
                    "gather_statistics": False,
                    "columns": self.columns_ext,
                }
                kwargs.update(self.kwargs)
                reader = dd.read_parquet if self.cpu else dask_cudf.read_parquet
            elif self.kind_ext == "csv":
                # Read from CSV dataset
                kwargs = {"usecols": self.columns_ext}
                kwargs.update(self.kwargs)
                reader = dd.read_csv if self.cpu else dask_cudf.read_csv
            else:
                raise ValueError("Disk format not yet supported")
            _ext = reader(self.df_ext, **kwargs)

        # Take subset of columns if a list is specified
        if self.columns_ext:
            _ext = _ext[self.columns_ext]

        # Drop duplicates if requested
        if self.drop_duplicates_ext:
            if isinstance(_ext, dd.DataFrame):
                return _ext.drop_duplicates(ignore_index=True)
            _ext.drop_duplicates(ignore_index=True, inplace=True)

        return _ext


    def _merge(self, df, _ext):
        if isinstance(_ext, dd.DataFrame):
            _ddf = dd.from_pandas(df, npartitions=1)
            return _ddf.merge(_ext, left_on=self.on, right_on=self.on_ext, how=self.how).compute()
        else:
            return df.merge(_ext, left_on=self.on, right_on=self.on_ext, how=self.how)


    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        self.cpu = not isinstance(df, cudf.DataFrame)
        tmp = "__tmp__"  # Temporary column for sorting
        df[tmp] = _arange(len(df), like_df=df, dtype="int32")
        new_df = self._merge(df, self._ext)
        new_df = new_df.sort_values(tmp)
        new_df.drop(columns=[tmp], inplace=True)
        df.drop(columns=[tmp], inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    def output_column_names(self, columns):
        if self.columns_ext:
            return list(set(columns + self.columns_ext))
        return list(set(columns + list(self._ext.columns)))


def _check_partition_count(df):
    if hasattr(df, "npartitions"):
        if df.npartitions > 3:
            warnings.warn(
                f"Joining an external Dask collection with "
                f"{df.npartitions} partitions. This transformation "
                f"requires a broadcast merge, which can be problematic "
                f"when the external collection is too large."
            )
    return df


def _detect_format(data):
    """Utility to detect the format of `data`"""

    if isinstance(data, Dataset):
        return "dataset"
    elif isinstance(data, dd.DataFrame):
        if isinstance(data._meta, cudf.DataFrame):
            return "dask-cudf"
        return "dask-dataframe"
    elif isinstance(data, cudf.DataFrame):
        return "cudf"
    elif isinstance(data, pd.DataFrame):
        return "pandas"
    elif isinstance(data, pa.Table):
        return "arrow"
    else:
        if isinstance(data, list) and data:
            file_type = str(data[0]).split(".")[-1]
        else:
            file_type = str(data).split(".")[-1]
        if file_type not in ("parquet", "csv"):
            raise ValueError("Data format not recognized.")
        return file_type
