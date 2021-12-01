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
from dask.delayed import Delayed

from nvtabular.dispatch import (
    DataFrameType,
    _arange,
    _concat_columns,
    _random_state,
    _read_parquet_dispatch,
)
from nvtabular.graph.tags import Tags

from . import categorify as nvt_cat
from .moments import _custom_moments
from .operator import ColumnSelector, Operator
from .stat_operator import StatOperator

CATEGORICAL = Tags.CATEGORICAL


class TargetEncoding(StatOperator):
    """
    Target encoding is a common feature-engineering technique for
    categorical columns in tabular datasets. For each categorical group,
    the mean of a continuous target column is calculated, and the
    group-specific mean of each row is used to create a new feature (column).
    To prevent overfitting, the following additional logic is applied:

        1. Cross Validation: To prevent overfitting in training data,
        a cross-validation strategy is used - The data is split into
        k random "folds", and the mean values within the i-th fold are
        calculated with data from all other folds. The cross-validation
        strategy is only employed when the dataset is used to update
        recorded statistics. For transformation-only workflow execution,
        global-mean statistics are used instead.

        2. Smoothing: To prevent overfitting for low cardinality categories,
        the means are smoothed with the overall mean of the target variable.

    Target Encoding Function::

        TE = ((mean_cat*count_cat)+(mean_global*p_smooth)) / (count_cat+p_smooth)

        count_cat := count of the categorical value
        mean_cat := mean target value of the categorical value
        mean_global := mean target value of the whole dataset
        p_smooth := smoothing factor

    Example usage::

        # First, we can transform the label columns to binary targets
        LABEL_COLUMNS = ['label1', 'label2']
        labels = ColumnSelector(LABEL_COLUMNS) >> (lambda col: (col>0).astype('int8'))
        # We target encode cat1, cat2 and the cross columns cat1 x cat2
        target_encode = (
            ['cat1', 'cat2', ['cat2','cat3']] >>
            nvt.ops.TargetEncoding(
                labels,
                kfold=5,
                p_smooth=20,
                out_dtype="float32",
                )
        )
        processor = nvt.Workflow(target_encode)

    Parameters
    -----------
    target : str
        Continuous target column to use for the encoding of cat_groups.
        The same continuous target will be used for all `cat_groups`.
    target_mean : float
        Global mean of the target column to use for encoding.
        Supplying this value up-front will improve performance.
    kfold : int, default 3
        Number of cross-validation folds to use while gathering statistics.
    fold_seed : int, default 42
        Random seed to use for numpy-based fold assignment.
    p_smooth : int, default 20
        Smoothing factor.
    out_col : str or list of str, default is problem-specific
        Name of output target-encoding column. If `cat_groups` includes
        multiple elements, this should be a list of the same length (and
        elements must be unique).
    out_dtype : str, default is problem-specific
        dtype of output target-encoding columns.
    tree_width : dict or int, optional
        Tree width of the hash-based groupby reduction for each categorical
        column. High-cardinality columns may require a large `tree_width`,
        while low-cardinality columns can likely use `tree_width=1`.
        If passing a dict, each key and value should correspond to the column
        name and width, respectively. The default value is 8 for all columns.
    cat_cache : {"device", "host", "disk"} or dict
        Location to cache the list of unique categories for
        each categorical column. If passing a dict, each key and value
        should correspond to the column name and location, respectively.
        Default is "host" for all columns.
    out_path : str, optional
        Root directory where category statistics will be written out in
        parquet format.
    on_host : bool, default True
        Whether to convert cudf data to pandas between tasks in the hash-based
        groupby reduction. The extra host <-> device data movement can reduce
        performance.  However, using `on_host=True` typically improves stability
        (by avoiding device-level memory pressure).
    name_sep : str, default "_"
        String separator to use between concatenated column names
        for multi-column groups.
    drop_folds : bool, default True
        Whether to drop the "__fold__" column created. This is really only useful for unittests.
    """

    def __init__(
        self,
        target,
        target_mean=None,
        kfold=None,
        fold_seed=42,
        p_smooth=20,
        out_col=None,
        out_dtype=None,
        tree_width=None,
        cat_cache="host",
        out_path=None,
        on_host=True,
        name_sep="_",
        drop_folds=True,
    ):
        super().__init__()

        self.target = [target] if isinstance(target, str) else target
        self.dependency = self.target

        self.target_mean = target_mean
        self.kfold = kfold or 3
        self.fold_seed = fold_seed
        self.p_smooth = p_smooth
        self.out_col = [out_col] if isinstance(out_col, str) else out_col
        self.out_dtype = out_dtype
        self.tree_width = tree_width
        self.out_path = out_path or "./"
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.name_sep = name_sep
        self.drop_folds = drop_folds
        self.fold_name = "__fold__"
        self.stats = {}
        self.means = {}  # TODO: just update target_mean?

    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame):
        moments = None
        self.target = self.target.names if isinstance(self.target, ColumnSelector) else self.target
        if self.target_mean is None:
            # calculate the mean if we don't have it already
            moments = _custom_moments(ddf[self.target])

        col_groups = col_selector.grouped_names

        if self.kfold > 1:
            # Add new fold column if necessary
            if self.fold_name not in ddf.columns:
                ddf[self.fold_name] = ddf.index.map_partitions(
                    _add_fold,
                    self.kfold,
                    self.fold_seed,
                    meta=_add_fold(ddf._meta.index, self.kfold, self.fold_seed),
                )

            # Add new col_groups with fold
            for group in col_selector.grouped_names:
                if isinstance(group, tuple):
                    group = list(group)
                if isinstance(group, list):
                    col_groups.append([self.fold_name] + group)
                else:
                    col_groups.append([self.fold_name, group])

        dsk, key = nvt_cat._category_stats(
            ddf,
            nvt_cat.FitOptions(
                col_groups,
                self.target,
                ["count", "sum"],
                self.out_path,
                0,
                self.tree_width,
                self.on_host,
                concat_groups=False,
                name_sep=self.name_sep,
            ),
        )
        return Delayed(key, dsk), moments

    def fit_finalize(self, dask_stats):
        for col, value in dask_stats[0].items():
            self.stats[col] = value
        for col in dask_stats[1].index:
            self.means[col] = float(dask_stats[1]["mean"].loc[col])

    def dependencies(self):
        return self.dependency

    def output_column_names(self, columns):
        if hasattr(self.target, "output_columns"):
            self.target = self.target.output_columns

        ret = []
        for cat in columns.grouped_names:
            cat = [cat] if isinstance(cat, str) else cat
            ret.extend(self._make_te_name(cat.names if isinstance(cat, ColumnSelector) else cat))

        if self.kfold > 1 and not self.drop_folds:
            ret.append(self.fold_name)

        return ColumnSelector(ret)

    def set_storage_path(self, new_path, copy=False):
        self.stats = nvt_cat._copy_storage(self.stats, self.out_path, new_path, copy)
        self.out_path = new_path

    def clear(self):
        self.stats = {}
        self.means = {}

    def _make_te_name(self, cat_group):
        tag = nvt_cat._make_name(*cat_group, sep=self.name_sep)
        target = self.target
        if isinstance(self.target, list):
            target = ColumnSelector(self.target)
        return [f"TE_{tag}_{x}" for x in target.names]

    def _op_group_logic(self, cat_group, df, y_mean, fit_folds, group_ind):
        # Define name of new TE column
        if isinstance(self.out_col, list):
            if group_ind >= len(self.out_col):
                raise ValueError("out_col and cat_groups are different sizes.")
            out_col = self.out_col[group_ind]
            out_col = [out_col] if isinstance(out_col, str) else out_col
            # ToDo Test
            if len(out_col) != len(self.target):
                raise ValueError("out_col and target are different sizes.")
        else:
            out_col = self._make_te_name(cat_group)

        # Initialize new data
        _read_pq_func = _read_parquet_dispatch(df)
        tmp = "__tmp__"

        if fit_folds:
            # Groupby Aggregation for each fold
            cols = ["__fold__"] + cat_group
            storage_name_folds = nvt_cat._make_name(*cols, sep=self.name_sep)
            path_folds = self.stats[storage_name_folds]
            agg_each_fold = nvt_cat._read_groupby_stat_df(
                path_folds, storage_name_folds, self.cat_cache, _read_pq_func
            )
            agg_each_fold.columns = cols + ["count_y"] + [x + "_sum_y" for x in self.target]
        else:
            cols = cat_group

        # Groupby Aggregation for all data
        storage_name_all = nvt_cat._make_name(*cat_group, sep=self.name_sep)
        path_all = self.stats[storage_name_all]
        agg_all = nvt_cat._read_groupby_stat_df(
            path_all, storage_name_all, self.cat_cache, _read_pq_func
        )
        agg_all.columns = cat_group + ["count_y_all"] + [x + "_sum_y_all" for x in self.target]

        if fit_folds:
            agg_each_fold = agg_each_fold.merge(agg_all, on=cat_group, how="left")
            agg_each_fold["count_y_all"] = agg_each_fold["count_y_all"] - agg_each_fold["count_y"]
            for i, x in enumerate(self.target):
                agg_each_fold[x + "_sum_y_all"] = (
                    agg_each_fold[x + "_sum_y_all"] - agg_each_fold[x + "_sum_y"]
                )
                agg_each_fold[out_col[i]] = (
                    agg_each_fold[x + "_sum_y_all"] + self.p_smooth * y_mean[x]
                ) / (agg_each_fold["count_y_all"] + self.p_smooth)

            agg_each_fold = agg_each_fold.drop(
                ["count_y_all", "count_y"]
                + [x + "_sum_y" for x in self.target]
                + [x + "_sum_y_all" for x in self.target],
                axis=1,
            )
            tran_df = df[cols + [tmp]].merge(agg_each_fold, on=cols, how="left")
            del agg_each_fold
        else:
            for i, x in enumerate(self.target):
                agg_all[out_col[i]] = (agg_all[x + "_sum_y_all"] + self.p_smooth * y_mean[x]) / (
                    agg_all["count_y_all"] + self.p_smooth
                )
            agg_all = agg_all.drop(
                ["count_y_all"] + [x + "_sum_y_all" for x in self.target], axis=1
            )
            tran_df = df[cols + [tmp]].merge(agg_all, on=cols, how="left")
            del agg_all

        # TODO: There is no need to perform the `agg_each_fold.merge(agg_all, ...)` merge
        #     for every partition.  We can/should cache the result for better performance.

        for i, x in enumerate(self.target):
            tran_df[out_col[i]] = tran_df[out_col[i]].fillna(y_mean[x])
        if self.out_dtype is not None:
            tran_df[out_col] = tran_df[out_col].astype(self.out_dtype)

        tran_df = tran_df.sort_values(tmp, ignore_index=True)
        tran_df.drop(columns=cols + [tmp], inplace=True)

        # Make sure we are preserving the index of df
        tran_df.index = df.index

        return tran_df

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        # Add temporary column for sorting
        tmp = "__tmp__"
        df[tmp] = _arange(len(df), like_df=df, dtype="int32")

        fit_folds = self.kfold > 1
        if fit_folds:
            df[self.fold_name] = _add_fold(df.index, self.kfold, self.fold_seed)

        # Need mean of contiuous target column
        y_mean = self.target_mean or self.means

        # Loop over categorical-column groups and apply logic
        new_df = None
        for ind, cat_group in enumerate(col_selector.grouped_names):
            if isinstance(cat_group, tuple):
                cat_group = list(cat_group)
            elif isinstance(cat_group, str):
                cat_group = [cat_group]

            if new_df is None:
                new_df = self._op_group_logic(cat_group, df, y_mean, fit_folds, ind)
            else:
                _df = self._op_group_logic(cat_group, df, y_mean, fit_folds, ind)
                new_df = _concat_columns([new_df, _df])

        # Drop temporary columns
        df.drop(columns=[tmp, "__fold__"] if fit_folds and self.drop_folds else [tmp], inplace=True)
        if fit_folds and not self.drop_folds:
            new_df[self.fold_name] = df[self.fold_name]
        return new_df

    def _get_tags(self):
        return [CATEGORICAL]

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__


def _add_fold(s, kfold, fold_seed=None):
    """Deterministically computes a '__fold__' column, given an optional
    random seed"""
    typ = np.min_scalar_type(kfold * 2)
    if fold_seed is None:
        # If we don't have a specific seed,
        # just use a simple modulo-based mapping
        fold = _arange(len(s), like_df=s, dtype=typ)
        np.mod(fold, kfold, out=fold)
        return fold
    else:
        state = _random_state(fold_seed, like_df=s)
        return state.choice(_arange(kfold, like_df=s, dtype=typ), len(s))
