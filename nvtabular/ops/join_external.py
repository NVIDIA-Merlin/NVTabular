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

from nvtabular.worker import fetch_table_data, get_worker_cache

from .operator import Operator


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
    df_ext : DataFrame, pyarrow.Table, or file path
        The external table to join to each partition of the dataset.
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
        if self.how not in ("left", "inner"):
            raise ValueError("Only left join is currently supported.")
        if self.kind_ext not in ("arrow", "cudf", "pandas", "parquet", "csv"):
            raise ValueError("kind_ext option not recognized.")

    @property
    @functools.lru_cache(1)
    def _ext(self):
        # Define _ext, depending on `kind_ext`
        if self.kind_ext == "cudf":
            _ext = self.df_ext
        elif self.kind_ext == "pandas":
            _ext = cudf.DataFrame.from_pandas(self.df_ext)
        elif self.kind_ext == "arrow":
            _ext = cudf.DataFrame.from_arrow(self.df_ext)
        else:
            if self.kind_ext == "parquet":
                reader = cudf.read_parquet
            elif self.kind_ext == "csv":
                reader = cudf.read_csv
            else:
                raise ValueError("Disk format not yet supported")

            with get_worker_cache(self.df_ext) as cached_table:
                _ext = fetch_table_data(
                    cached_table,
                    self.df_ext,
                    cache=self.cache,
                    columns=self.columns_ext,
                    reader=reader,
                    **self.kwargs,
                )

        # Take subset of columns if a list is specified
        if self.columns_ext:
            _ext = _ext[self.columns_ext]

        # Drop duplicates if requested
        if self.drop_duplicates_ext:
            _ext.drop_duplicates(ignore_index=True, inplace=True)

        return _ext

    def transform(self, columns, gdf: cudf.DataFrame) -> cudf.DataFrame:
        tmp = "__tmp__"  # Temporary column for sorting
        gdf[tmp] = cupy.arange(len(gdf), dtype="int32")
        new_gdf = gdf.merge(self._ext, left_on=self.on, right_on=self.on_ext, how=self.how)
        new_gdf = new_gdf.sort_values(tmp)
        new_gdf.drop(columns=[tmp], inplace=True)
        gdf.drop(columns=[tmp], inplace=True)
        new_gdf.reset_index(drop=True, inplace=True)
        return new_gdf

    transform.__doc__ = Operator.transform.__doc__

    def output_column_names(self, columns):
        if self.columns_ext:
            return list(set(columns + self.columns_ext))
        return list(set(columns + list(self._ext.columns)))


def _detect_format(data):
    """Utility to detect the format of `data`"""

    if isinstance(data, cudf.DataFrame):
        return "cudf"
    elif isinstance(data, pd.DataFrame):
        return "pandas"
    elif isinstance(data, pa.Table):
        return "arrow"
    else:
        file_type = str(data).split(".")[-1]
        if file_type not in ("parquet", "csv"):
            raise ValueError("Data format not recognized.")
        return file_type
