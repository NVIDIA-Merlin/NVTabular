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

import dask.dataframe as dd
import pandas as pd
from dask.delayed import Delayed

import nvtabular as nvt
from nvtabular.dispatch import DataFrameType, _arange, _concat_columns, _read_parquet_dispatch

from . import categorify as nvt_cat
from .operator import ColumnNames, Operator
from .stat_operator import StatOperator


class JoinGroupby(StatOperator):
    """
    One of the ways to create new features is to calculate
    the basic statistics of the data that is grouped by categorical
    features. This operator groups the data by the given categorical
    feature(s) and calculates the desired statistics of requested continuous
    features (along with the count of rows in each group). The aggregated
    statistics are merged with the data (by joining on the desired
    categorical columns).

    Example usage::

        # Use JoinGroupby to define a NVTabular workflow
        groupby_features = ['cat1', 'cat2', 'cat3'] >> ops.JoinGroupby(
            out_path=str(tmpdir), stats=['sum','count'], cont_cols=['num1']
        )
        processor = nvtabular.Workflow(groupby_features)

    Parameters
    -----------
    cont_cols : list of str or ColumnGroup
        The continuous columns to calculate statistics for
        (for each unique group in each column in `columns`).
    stats : list of str, default []
        List of statistics to calculate for each unique group. Note
        that "count" corresponds to the group itself, while all
        other statistics correspond to a specific continuous column.
        Supported statistics include ["count", "sum", "mean", "std", "var"].
    tree_width : dict or int, optional
        Tree width of the hash-based groupby reduction for each categorical
        column. High-cardinality columns may require a large `tree_width`,
        while low-cardinality columns can likely use `tree_width=1`.
        If passing a dict, each key and value should correspond to the column
        name and width, respectively. The default value is 8 for all columns.
    cat_cache: ToDo Describe
        TEXT
    out_path : str, optional
        Root directory where groupby statistics will be written out in
        parquet format.
    on_host : bool, default True
        Whether to convert cudf data to pandas between tasks in the hash-based
        groupby reduction. The extra host <-> device data movement can reduce
        performance.  However, using `on_host=True` typically improves stability
        (by avoiding device-level memory pressure).
    name_sep : str, default "_"
        String separator to use between concatenated column names
        for multi-column groups.
    """

    def __init__(
        self,
        cont_cols=None,
        stats=("count",),
        tree_width=None,
        cat_cache="host",
        out_path=None,
        on_host=True,
        name_sep="_",
    ):
        super().__init__()

        self.storage_name = {}
        self.name_sep = name_sep
        self.cont_cols = (
            cont_cols if isinstance(cont_cols, nvt.ColumnGroup) else nvt.ColumnGroup(cont_cols)
        )
        self.cont_names = self.cont_cols.columns
        self.stats = stats
        self.tree_width = tree_width
        self.out_path = out_path or "./"
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.categories = {}

        supported_ops = ["count", "sum", "mean", "std", "var", "min", "max"]
        for op in self.stats:
            if op not in supported_ops:
                raise ValueError(op + " operation is not supported.")

    def fit(self, columns: ColumnNames, ddf: dd.DataFrame):
        if isinstance(columns, list):
            for group in columns:
                if isinstance(group, (list, tuple)) and len(group) > 1:
                    name = nvt_cat._make_name(*group, sep=self.name_sep)
                    for col in group:
                        self.storage_name[col] = name

        # Check metadata type to reset on_host and cat_cache if the
        # underlying ddf is already a pandas-backed collection
        if isinstance(ddf._meta, pd.DataFrame):
            self.on_host = False
            # Cannot use "device" caching if the data is pandas-backed
            self.cat_cache = "host" if self.cat_cache == "device" else self.cat_cache

        dsk, key = nvt_cat._category_stats(
            ddf,
            nvt_cat.FitOptions(
                columns,
                self.cont_names,
                self.stats,
                self.out_path,
                0,
                self.tree_width,
                self.on_host,
                concat_groups=False,
                name_sep=self.name_sep,
            ),
        )
        return Delayed(key, dsk)

    def fit_finalize(self, dask_stats):
        for col in dask_stats:
            self.categories[col] = dask_stats[col]

    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        new_df = type(df)()
        tmp = "__tmp__"  # Temporary column for sorting
        df[tmp] = _arange(len(df), like_df=df, dtype="int32")

        cat_names, multi_col_group = nvt_cat._get_multicolumn_names(
            columns, df.columns, self.name_sep
        )

        _read_pq_func = _read_parquet_dispatch(df)
        for name in cat_names:
            new_part = type(df)()
            storage_name = self.storage_name.get(name, name)
            name = multi_col_group.get(name, name)
            path = self.categories[storage_name]
            selection_l = list(name) if isinstance(name, tuple) else [name]
            selection_r = list(name) if isinstance(name, tuple) else [storage_name]

            stat_df = nvt_cat._read_groupby_stat_df(
                path, storage_name, self.cat_cache, _read_pq_func
            )
            tran_df = df[selection_l + [tmp]].merge(
                stat_df, left_on=selection_l, right_on=selection_r, how="left"
            )
            tran_df = tran_df.sort_values(tmp)
            tran_df.drop(columns=selection_l + [tmp], inplace=True)
            new_cols = [c for c in tran_df.columns if c not in new_df.columns]
            new_part = tran_df[new_cols].reset_index(drop=True)
            new_df = _concat_columns([new_df, new_part])
        df.drop(columns=[tmp], inplace=True)
        return new_df

    def dependencies(self):
        return self.cont_cols

    def output_column_names(self, columns):
        # TODO: the names here are defined in categorify/mid_level_groupby
        # refactor to have a common implementation
        output = []
        for name in columns:
            if isinstance(name, (tuple, list)):
                name = nvt_cat._make_name(*name, sep=self.name_sep)
            for cont in self.cont_names:
                for stat in self.stats:
                    if stat == "count":
                        output.append(f"{name}_{stat}")
                    else:
                        output.append(f"{name}_{cont}_{stat}")
        return output

    def set_storage_path(self, new_path, copy=False):
        self.categories = nvt_cat._copy_storage(self.categories, self.out_path, new_path, copy)
        self.out_path = new_path

    def clear(self):
        self.categories = {}
        self.storage_name = {}

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
