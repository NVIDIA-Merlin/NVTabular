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
import cudf
import cupy
from dask.core import flatten

from . import categorify as nvt_cat
from .groupby_statistics import GroupbyStatistics
from .operator import CAT
from .transform_operator import DFOperator


class JoinGroupby(DFOperator):
    """
    One of the ways to create new features is to calculate
    the basic statistics of the data that is grouped by categorical
    features. This operator groups the data by the given categorical
    feature(s) and calculates the desired statistics of requested continuous
    features (along with the count of rows in each group). The aggregated
    statistics are merged with the data (by joining on the desired
    categorical columns).

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add JoinGroupby to the workflow
        proc.add_feature(
            JoinGroupby(
                columns=['cat1', 'cat2', 'cat3'], # columns which are groupby
                cont_names=['num1'], # continuous column, which the statistics are applied to
                stats=['sum','count']), # statistics, which are applied
        )

    Parameters
    -----------
    cont_names : list of str
        The continuous column names to calculate statistics for
        (for each unique group in each column in `columns`).
    stats : list of str, default []
        List of statistics to calculate for each unique group. Note
        that "count" corresponds to the group itself, while all
        other statistics correspond to a specific continuous column.
        Supported statistics include ["count", "sum", "mean", "std", "var"].
    columns : list of str or list(str), default None
        Categorical columns (or multi-column "groups") to target for this op.
        If None, the operation will target all known categorical columns.
    tree_width : dict or int, optional
        Passed to `GroupbyStatistics` dependency.
    out_path : str, optional
        Passed to `GroupbyStatistics` dependency.
    on_host : bool, default True
        Passed to `GroupbyStatistics` dependency.
    name_sep : str, default "_"
        String separator to use between concatenated column names
        for multi-column groups.
    """

    default_in = CAT
    default_out = CAT

    def __init__(
        self,
        cont_names=None,
        stats=["count"],
        columns=None,
        tree_width=None,
        cat_cache="host",
        out_path=None,
        on_host=True,
        name_sep="_",
        stat_name=None,
    ):
        self.column_groups = None
        self.storage_name = {}
        self.name_sep = name_sep
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, list):
            self.column_groups = columns
            columns = list(set(flatten(columns, container=list)))
            for group in self.column_groups:
                if isinstance(group, list) and len(group) > 1:
                    name = nvt_cat._make_name(*group, sep=self.name_sep)
                    for col in group:
                        self.storage_name[col] = name

        super().__init__(columns=columns, replace=False)
        self.cont_names = cont_names
        self.stats = stats
        self.tree_width = tree_width
        self.out_path = out_path
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.stat_name = stat_name or "gb_categories"

    @property
    def req_stats(self):
        return [
            GroupbyStatistics(
                columns=self.column_groups or self.columns,
                concat_groups=False,
                cont_names=self.cont_names,
                stats=self.stats,
                tree_width=self.tree_width,
                out_path=self.out_path,
                on_host=self.on_host,
                stat_name=self.stat_name,
                name_sep=self.name_sep,
            )
        ]

    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):

        new_gdf = cudf.DataFrame()
        tmp = "__tmp__"  # Temporary column for sorting
        gdf[tmp] = cupy.arange(len(gdf), dtype="int32")
        if self.column_groups:
            cat_names, multi_col_group = nvt_cat._get_multicolumn_names(
                self.column_groups, gdf.columns, self.name_sep
            )
        else:
            multi_col_group = {}
            cat_names = [name for name in target_columns if name in gdf.columns]

        for name in cat_names:
            storage_name = self.storage_name.get(name, name)
            name = multi_col_group.get(name, name)
            path = stats_context[self.stat_name][storage_name]
            selection_l = name.copy() if isinstance(name, list) else [name]
            selection_r = name if isinstance(name, list) else [storage_name]

            stat_gdf = nvt_cat._read_groupby_stat_df(path, storage_name, self.cat_cache)
            tran_gdf = gdf[selection_l + [tmp]].merge(
                stat_gdf, left_on=selection_l, right_on=selection_r, how="left"
            )
            tran_gdf = tran_gdf.sort_values(tmp)
            tran_gdf.drop(columns=selection_l + [tmp], inplace=True)
            new_cols = [c for c in tran_gdf.columns if c not in new_gdf.columns]
            new_gdf[new_cols] = tran_gdf[new_cols].reset_index(drop=True)
        gdf.drop(columns=[tmp], inplace=True)
        return new_gdf
