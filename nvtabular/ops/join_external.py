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
import warnings

try:
    import cudf
except ImportError:
    cudf = None

import dask.dataframe as dd
import pandas as pd

from merlin.core.dispatch import (
    DataFrameType,
    ExtData,
    arange,
    convert_data,
    create_merlin_dataset,
    detect_format,
    to_host,
)
from merlin.schema import Schema

from .operator import ColumnSelector, Operator


class JoinExternal(Operator):
    """
    Join each dataset partition to an external table. For performance
    reasons, only "left" and "inner" join transformations are supported.

    Example usage::

        # Load dataset which should be joined to the main dataset
        df_external = cudf.read_parquet('external.parquet')

        # Use JoinExternal to define a NVTabular workflow
        joined = ColumnSelector(columns_left) >> nvt.ops.JoinExternal(
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
    kind_ext : ExtData; Optional
        Format of ``df_ext``.  If nothing is specified, the format
        will be inferred.
    cache : {"device", "host", "disk"}
        Where to cache ``df_ext`` between transformations. Only used
        if the data is originally stored on disk. The "host" option
        is also supported when ``df_ext`` is a ``cudf.DataFrame``.
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
        self.df_ext = create_merlin_dataset(df_ext)
        self.on_ext = on_ext or self.on
        self.how = how
        self.kind_ext = kind_ext or detect_format(self.df_ext)
        self.columns_ext = columns_ext
        self.drop_duplicates_ext = drop_duplicates_ext
        self.cache = cache
        self.kwargs = kwargs
        self.cpu = None
        self._ext_cache = None
        if cudf is None:
            self.cpu = True
        if self.how not in ("left", "inner"):
            raise ValueError("Only left join is currently supported.")
        if not isinstance(self.kind_ext, ExtData):
            raise ValueError("kind_ext option not recognized.")
        super().__init__()

    @property
    def _ext(self):

        if self._ext_cache is not None:
            # Return cached result if present
            return convert_data(self._ext_cache, cpu=self.cpu)

        # Use Dataset.to_ddf
        _dataset = self.df_ext
        if self.cpu:
            _dataset.to_cpu()
        else:
            _dataset.to_gpu()
        _ext = _check_partition_count(_dataset.to_ddf(columns=self.columns_ext))

        # Take subset of columns if a list is specified
        if self.columns_ext:
            _ext = _ext[self.columns_ext]

        # Drop duplicates if requested
        if self.drop_duplicates_ext:
            if isinstance(_ext, dd.DataFrame):
                _ext = _ext.drop_duplicates(ignore_index=True)
            else:
                _ext.drop_duplicates(ignore_index=True, inplace=True)

        # Cache and return
        if self.cache == "host":
            self._ext_cache = to_host(_ext)
        elif self.cache == "device" or self.kind_ext not in (ExtData.PARQUET, ExtData.CSV):
            self._ext_cache = _ext
        return _ext

    def _merge(self, df, _ext):
        if isinstance(_ext, dd.DataFrame):
            _ddf = dd.from_pandas(df, npartitions=1)
            return _ddf.merge(_ext, left_on=self.on, right_on=self.on_ext, how=self.how).compute()
        else:
            return df.merge(_ext, left_on=self.on, right_on=self.on_ext, how=self.how)

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        self.cpu = isinstance(df, pd.DataFrame)
        tmp = "__tmp__"  # Temporary column for sorting
        df[tmp] = arange(len(df), like_df=df, dtype="int32")
        new_df = self._merge(df, self._ext)
        new_df = new_df.sort_values(tmp)
        new_df.drop(columns=[tmp], inplace=True)
        df.drop(columns=[tmp], inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        self._validate_matching_cols(input_schema, parents_selector, "computing input selector")
        return parents_selector

    def compute_output_schema(self, input_schema, col_selector, prev_output_schema=None):
        # must load in the schema from the external dataset
        input_schema = input_schema + self.df_ext.schema
        return super().compute_output_schema(input_schema, col_selector, prev_output_schema)

    def column_mapping(self, col_selector):
        column_mapping = {}
        ext_columns = self.columns_ext if self.columns_ext else self._ext.columns

        # This maintains the order which set() does not
        combined_col_names = dict.fromkeys(col_selector.names + list(ext_columns)).keys()

        for col_name in combined_col_names:
            column_mapping[col_name] = [col_name]

        return column_mapping

    def _compute_dtype(self, col_schema, input_schema):
        if col_schema.name in input_schema.column_names:
            return super()._compute_dtype(col_schema, input_schema)
        else:
            col_dtype = self.df_ext.schema.column_schemas[col_schema.name].dtype
            return col_schema.with_dtype(col_dtype)

    def _compute_tags(self, col_schema, input_schema):
        return col_schema

    def _compute_properties(self, col_schema, input_schema):
        return col_schema


def _check_partition_count(df):
    if hasattr(df, "npartitions"):
        if df.npartitions == 1:
            # Materialize single-partition collections
            return df.compute()
        if df.npartitions > 3:
            warnings.warn(
                f"Joining an external Dask collection with "
                f"{df.npartitions} partitions. This transformation "
                f"requires a broadcast merge, which can be problematic "
                f"when the external collection is too large."
            )
    return df
