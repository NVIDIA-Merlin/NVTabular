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

from . import categorify as nvt_cat
from .groupby_statistics import GroupbyStatistics
from .moments import Moments
from .operator import ALL
from .transform_operator import DFOperator


class TargetEncoding(DFOperator):
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

    Function:
    TE = ((mean_cat*count_cat)+(mean_global*p_smooth)) / (count_cat+p_smooth)

    count_cat := count of the categorical value
    mean_cat := mean of target value for the categorical value
    mean_global := mean of the target value in the dataset
    p_smooth := smoothing factor

    Parameters
    -----------
    cat_group : list of str
        Column, or group of columns, to target encode.
    cont_target : str
        Continuous target column to use for the encoding of cat_group.
    kfold : int, default 3
        Numbner of cross-validation folds to use while gathering
        statistics (during `GroupbyStatistics`).
    fold_seed : int, default 42
        Random seed to use for cupy-based fold assignment.
    drop_folds : bool, default True
        Whether to drop the "__fold__" column created by the
        `GroupbyStatistics` dependency (after the transformation).
    p_smooth : int, default 20
        Smoothing factor.
    out_col : str, default is problem-specific
        Name of output target-encoding column.
    out_dtype : str, default is problem-specific
        dtype of output target-encoding column.
    replace : bool, default False
        This parameter is ignored
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

    default_in = ALL
    default_out = ALL

    def __init__(
        self,
        cat_group,
        cont_target,
        kfold=None,
        fold_seed=42,
        p_smooth=20,
        out_col=None,
        out_dtype=None,
        replace=False,
        tree_width=None,
        cat_cache="host",
        out_path=None,
        on_host=True,
        name_sep="_",
        stat_name=None,
        drop_folds=True,
    ):
        super().__init__(replace=replace)
        self.cat_group = cat_group if isinstance(cat_group, list) else [cat_group]
        self.cont_target = cont_target
        self.kfold = kfold or 3
        self.fold_seed = fold_seed
        self.p_smooth = p_smooth
        self.out_col = out_col
        self.out_dtype = out_dtype
        self.tree_width = tree_width
        self.out_path = out_path
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.name_sep = name_sep
        self.drop_folds = drop_folds
        self.stat_name = stat_name or "te_stats"

    @property
    def req_stats(self):
        return [
            Moments(columns=[self.cont_target]),
            GroupbyStatistics(
                columns=[self.cat_group],
                concat_groups=False,
                cont_names=[self.cont_target],
                stats=["count", "sum"],
                tree_width=self.tree_width,
                out_path=self.out_path,
                on_host=self.on_host,
                stat_name=self.stat_name,
                name_sep=self.name_sep,
                kfold=self.kfold,
                fold_seed=self.fold_seed,
                fold_groups=[self.cat_group],
            ),
        ]

    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):

        new_gdf = cudf.DataFrame()
        tmp = "__tmp__"  # Temporary column for sorting
        gdf[tmp] = cupy.arange(len(gdf), dtype="int32")

        if self.out_col is None:
            tag = nvt_cat._make_name(*self.cat_group, sep=self.name_sep)
            self.out_col = f"TE_{tag}_{self.cont_target}"

        # Need mean of contiuous target column
        y_mean = stats_context["means"][self.cont_target]

        # Only perform "fit" if fold column is present
        fit_folds = "__fold__" in gdf.columns

        if fit_folds:
            # Groupby Aggregation for each fold
            cols = ["__fold__"] + self.cat_group
            storage_name_folds = nvt_cat._make_name(*cols, sep=self.name_sep)
            path_folds = stats_context[self.stat_name][storage_name_folds]
            agg_each_fold = nvt_cat._read_groupby_stat_df(
                path_folds, storage_name_folds, self.cat_cache
            )
            agg_each_fold.columns = cols + ["count_y", "sum_y"]
        else:
            cols = self.cat_group

        # Groupby Aggregation for all data
        storage_name_all = nvt_cat._make_name(*self.cat_group, sep=self.name_sep)
        path_all = stats_context[self.stat_name][storage_name_all]
        agg_all = nvt_cat._read_groupby_stat_df(path_all, storage_name_all, self.cat_cache)
        agg_all.columns = self.cat_group + ["count_y_all", "sum_y_all"]

        if fit_folds:
            agg_each_fold = agg_each_fold.merge(agg_all, on=self.cat_group, how="left")
            agg_each_fold["count_y_all"] = agg_each_fold["count_y_all"] - agg_each_fold["count_y"]
            agg_each_fold["sum_y_all"] = agg_each_fold["sum_y_all"] - agg_each_fold["sum_y"]
            agg_each_fold[self.out_col] = (agg_each_fold["sum_y_all"] + self.p_smooth * y_mean) / (
                agg_each_fold["count_y_all"] + self.p_smooth
            )
            agg_each_fold = agg_each_fold.drop(
                ["count_y_all", "count_y", "sum_y_all", "sum_y"], axis=1
            )
            tran_gdf = gdf[cols + [tmp]].merge(agg_each_fold, on=cols, how="left")
            del agg_each_fold
        else:
            agg_all[self.out_col] = (agg_all["sum_y_all"] + self.p_smooth * y_mean) / (
                agg_all["count_y_all"] + self.p_smooth
            )
            agg_all = agg_all.drop(["count_y_all", "sum_y_all"], axis=1)
            tran_gdf = gdf[cols + [tmp]].merge(agg_all, on=cols, how="left")
            del agg_all

        # TODO: There is no need to perform the `agg_each_fold.merge(agg_all, ...)` merge
        #     for every partition.  We can/should cache the result for better performance.

        tran_gdf[self.out_col] = tran_gdf[self.out_col].fillna(y_mean)
        if self.out_dtype is not None:
            tran_gdf[self.out_col] = tran_gdf[self.out_col].astype(self.out_dtype)

        tran_gdf = tran_gdf.sort_values(tmp, ignore_index=True)
        tran_gdf.drop(columns=cols + [tmp], inplace=True)
        new_cols = [c for c in tran_gdf.columns if c not in new_gdf.columns]
        new_gdf[new_cols] = tran_gdf[new_cols]

        # Make sure we are preserving the index of gdf
        new_gdf.index = gdf.index

        gdf.drop(
            columns=[tmp, "__fold__"] if fit_folds and self.drop_folds else [tmp], inplace=True
        )
        return new_gdf
