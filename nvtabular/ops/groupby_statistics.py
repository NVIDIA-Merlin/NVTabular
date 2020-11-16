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
import cupy
import numpy as np
from dask.core import flatten
from dask.delayed import Delayed

from . import categorify as nvt_cat
from .stat_operator import StatOperator


class GroupbyStatistics(StatOperator):
    """
    Uses groupby aggregation to determine the unique groups of a categorical
    feature and calculates the desired statistics of requested continuous
    features (along with the count of rows in each group).  The statistics
    for each category will be written to a distinct parquet file, and a
    dictionary of paths will be returned as the final "statistics".

    Parameters
    -----------
    cont_names : list of str
        The continuous column names to calculate statistics for
        (for each unique group in each column in `columns`)
    stats : list of str, default []
        List of statistics to calculate for each unique group. Note
        that "count" corresponds to the group itself, while all
        other statistics correspond to a specific continuous column.
        Supported statistics include ["count", "sum", "mean", "std", "var", "min", "max"].
    columns : list of str or list(str), default None
        Categorical columns (or "column groups") to collect statistics for.
        If None, the operation will target all known categorical columns.
    fold_groups : list, default None
        List of groups to to perform a groupby aggregation with an additional
        "fold" column (typically for cross-validation).
    concat_groups : bool, default False
        Applies only if there are list elements in the ``columns`` input. If True,
        the values within these column groups will be concatenated, and the
        new (temporary) columns will be used to perform the groupby.  The purpose of
        this option is to enable multiple columns to be label-encoded jointly.
        (see Categorify). Note that this option is only allowed for the "count"
        statistics (with cont_names == None).
    tree_width : dict or int, optional
        Tree width of the hash-based groupby reduction for each categorical
        column. High-cardinality columns may require a large `tree_width`,
        while low-cardinality columns can likely use `tree_width=1`.
        If passing a dict, each key and value should correspond to the column
        name and width, respectively. The default value is 8 for all columns.
    out_path : str, optional
        Root directory where groupby statistics will be written out in
        parquet format.
    freq_threshold : int, default 0
        Categories with a `count` statistic less than this number will
        be omitted from the `GroupbyStatistics` output.
    on_host : bool, default True
        Whether to convert cudf data to pandas between tasks in the hash-based
        groupby reduction. The extra host <-> device data movement can reduce
        performance.  However, using `on_host=True` typically improves stability
        (by avoiding device-level memory pressure).
    name_sep : str, default "_"
        String separator to use between concatenated column names
        for multi-column groups.
    fold_name : str, default "__fold__"
        Name of the fold column to use for all groups in `fold_groups`.
    fold_seed : int, default 42
        Random seed to use for cupy-based fold assignment.
    kfold : str, default 3
        Number of cross-validation folds to use for all groups in `fold_groups`.
    """

    def __init__(
        self,
        cont_names=None,
        stats=None,
        columns=None,
        fold_groups=None,
        tree_width=None,
        out_path=None,
        on_host=True,
        freq_threshold=None,
        stat_name=None,
        concat_groups=False,
        name_sep="_",
        fold_name="__fold__",
        fold_seed=42,
        kfold=None,
    ):
        # Set column_groups if the user has passed in a list of columns
        self.column_groups = None
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, list):
            self.column_groups = columns
            columns = list(set(flatten(columns, container=list)))

        # Add fold_groups to columns
        if fold_groups and kfold > 1:
            fold_groups = [fold_groups] if isinstance(fold_groups, str) else fold_groups
            columns = columns or []
            self.column_groups = self.column_groups or []
            for col in list(set(flatten(fold_groups, container=list))):
                if col not in columns:
                    columns.append(col)

        super(GroupbyStatistics, self).__init__(columns)
        self.cont_names = cont_names or []
        self.stats = stats or []
        self.categories = {}
        self.tree_width = tree_width or 8
        self.on_host = on_host
        self.freq_threshold = freq_threshold
        self.out_path = out_path or "./"
        self.stat_name = stat_name or "categories"
        self.op_name = "GroupbyStatistics-" + self.stat_name
        self.concat_groups = concat_groups
        self.name_sep = name_sep
        self.kfold = kfold or 3
        self.fold_name = fold_name
        self.fold_seed = fold_seed
        self.fold_groups = fold_groups

    @property
    def _id(self):
        c_id = self._id_set
        if not self._id_set:
            c_id = str(self.op_name)
        return c_id

    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        if self.column_groups is None:
            col_groups = self.get_columns(columns_ctx, input_cols, target_cols)
        else:
            col_groups = self.column_groups.copy()
        supported_ops = ["count", "sum", "mean", "std", "var", "min", "max"]
        for op in self.stats:
            if op not in supported_ops:
                raise ValueError(op + " operation is not supported.")

        if self.fold_groups and self.kfold > 1:
            # Add new fold column if necessary
            if self.fold_name not in ddf.columns:

                def _add_fold(s, kfold, fold_seed):
                    typ = np.min_scalar_type(kfold * 2)
                    if fold_seed is None:
                        # If we don't have a specific seed,
                        # just use a simple modulo-based mapping
                        fold = cupy.arange(len(s), dtype=typ)
                        cupy.mod(fold, kfold, out=fold)
                        return fold
                    else:
                        cupy.random.seed(fold_seed)
                        return cupy.random.choice(cupy.arange(kfold, dtype=typ), len(s))

                ddf[self.fold_name] = ddf.index.map_partitions(
                    _add_fold,
                    self.kfold,
                    self.fold_seed,
                    meta=_add_fold(ddf._meta.index, self.kfold, self.fold_seed),
                )

                # Specify to workflow that the ddf has been updated
                self._ddf_out = ddf

            # Add new col_groups with fold
            for group in self.fold_groups:
                if isinstance(group, list):
                    col_groups.append([self.fold_name] + group)
                else:
                    col_groups.append([self.fold_name, group])

            # Make sure concat
            if self.concat_groups:
                raise ValueError("cannot use concat_groups=True with folds.")

        agg_cols = self.cont_names
        agg_list = self.stats
        dsk, key = nvt_cat._category_stats(
            ddf,
            col_groups,
            agg_cols,
            agg_list,
            self.out_path,
            self.freq_threshold,
            self.tree_width,
            self.on_host,
            stat_name=self.stat_name,
            concat_groups=self.concat_groups,
            name_sep=self.name_sep,
        )
        return Delayed(key, dsk)

    def finalize(self, dask_stats):
        for col in dask_stats:
            self.categories[col] = dask_stats[col]

    def registered_stats(self):
        return [self.stat_name]

    def stats_collected(self):
        result = [(self.stat_name, self.categories)]
        return result

    def clear(self):
        self.categories = {}
        return
