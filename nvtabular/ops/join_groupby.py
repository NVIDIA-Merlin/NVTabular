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
import numpy as np
import pandas as pd
from dask.delayed import Delayed

import nvtabular as nvt
from merlin.core.dispatch import DataFrameType, arange, concat_columns, read_parquet_dispatch
from merlin.schema import Schema

from . import categorify as nvt_cat
from .operator import ColumnSelector, Operator
from .stat_operator import StatOperator

AGG_DTYPES = {
    "count": np.int32,
    "std": np.float32,
    "var": np.float32,
    "mean": np.float32,
}


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
    cont_cols : list of str or WorkflowNode
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
        self.stats = stats
        self.tree_width = tree_width
        self.out_path = out_path or "./"
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.categories = {}

        self._cont_names = None

        if isinstance(cont_cols, nvt.WorkflowNode):
            self.cont_cols = cont_cols
        elif isinstance(cont_cols, ColumnSelector):
            self.cont_cols = self._cont_names = cont_cols
        else:
            self.cont_cols = self._cont_names = ColumnSelector(cont_cols)

        supported_ops = ["count", "sum", "mean", "std", "var", "min", "max"]
        for op in self.stats:
            if op not in supported_ops:
                raise ValueError(op + " operation is not supported.")

    @property
    def cont_names(self):
        if self._cont_names:
            return self._cont_names
        elif self.cont_cols.output_schema:
            return self.cont_cols.output_columns
        else:
            raise RuntimeError(
                "Can't compute continuous columns used by `JoinGroupby` "
                "until `Workflow` is fit to dataset or schema."
            )

    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame):
        for group in col_selector.subgroups:
            if len(group.names) > 1:
                name = nvt_cat._make_name(*group.names, sep=self.name_sep)
                for col in group.names:
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
                col_selector,
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

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        new_df = type(df)()
        tmp = "__tmp__"  # Temporary column for sorting
        df[tmp] = arange(len(df), like_df=df, dtype="int32")

        cat_names = []
        multi_col_group = {}
        for col_name in col_selector.grouped_names:
            if isinstance(col_name, (list, tuple)):
                name = nvt_cat._make_name(*col_name, sep=self.name_sep)
                if name not in cat_names and all(col in df.columns for col in col_name):
                    cat_names.append(name)
                    multi_col_group[name] = col_name
            elif col_name in df.columns:
                cat_names.append(col_name)

        _read_pq_func = read_parquet_dispatch(df)
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
            for col in new_part.columns:
                for agg in list(AGG_DTYPES.keys()):
                    if col.endswith(f"{self.name_sep}{agg}"):
                        new_dtype = AGG_DTYPES.get(agg, new_part[col].dtype)
                        new_part[col] = new_part[col].astype(new_dtype)
            new_df = concat_columns([new_df, new_part])
        df.drop(columns=[tmp], inplace=True)
        return new_df

    @property
    def dependencies(self):
        return self.cont_cols

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        self._validate_matching_cols(input_schema, parents_selector, "computing input selector")
        return parents_selector

    def column_mapping(self, col_selector):
        column_mapping = {}
        for group in col_selector.grouped_names:
            if isinstance(group, (tuple, list)):
                name = nvt_cat._make_name(*group, sep=self.name_sep)
                group = [*group]
            else:
                name = group
                group = [group]

            for cont in self.cont_names.names:
                for stat in self.stats:
                    if stat == "count":
                        column_mapping[f"{name}_{stat}"] = [*group]
                    else:
                        column_mapping[f"{name}_{cont}_{stat}"] = [cont, *group]

        return column_mapping

    def _compute_dtype(self, col_schema, input_schema):
        new_schema = super()._compute_dtype(col_schema, input_schema)

        dtype = new_schema.dtype
        is_list = new_schema.is_list

        for agg in list(AGG_DTYPES.keys()):
            if col_schema.name.endswith(f"{self.name_sep}{agg}"):
                dtype = AGG_DTYPES.get(agg, dtype)
                is_list = False
                break

        return col_schema.with_dtype(dtype, is_list=is_list, is_ragged=is_list)

    def set_storage_path(self, new_path, copy=False):
        self.categories = nvt_cat._copy_storage(self.categories, self.out_path, new_path, copy)
        self.out_path = new_path

    def clear(self):
        self.categories = {}
        self.storage_name = {}

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
