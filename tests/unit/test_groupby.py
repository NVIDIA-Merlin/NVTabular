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

import glob

import cudf
import pytest

import nvtabular.groupby as groupby
from tests.conftest import allcols_csv, mycols_csv


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("gpu_mem_trans_use", [0.00000001, 0.5])
def test_groupby_fit_merge_fim(datasets, batch, dskey, gpu_mem_trans_use):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")

    grouby_stats = groupby.GroupByMomentsCal(
        col="name-string",
        col_count="x",
        cont_col=["x", "y"],
        stats=["count", "sum"],
        gpu_mem_trans_use=gpu_mem_trans_use,
    )

    grouby_stats.fit(df_expect[["name-string", "x", "y"]])
    grouby_stats.fit_finalize()
    new_fea = grouby_stats.merge(df_expect[["name-string", "x", "y"]])
    new_fea_host = new_fea.to_pandas()
    new_fea_sum_host = new_fea_host[["name-string_x_sum", "name-string_y_sum"]]
    new_fea_count_host = new_fea_host[["name-string_count"]]

    df_expect_pd = df_expect.to_pandas()
    groups_pd = df_expect_pd[["name-string", "x", "y"]].groupby(["name-string"])

    sums_pd = groups_pd.sum()
    sums_pd["name-string"] = sums_pd.index
    sums_pd = sums_pd.reset_index(drop=True)

    new_fea_pd = df_expect_pd.merge(sums_pd, on=["name-string"], how="left")
    new_fea_pd = new_fea_pd[["x_y", "y_y"]]

    error = new_fea_sum_host.subtract(new_fea_pd, axis=1)

    e_x_y = error["x_y"].abs()
    e_x_y = e_x_y[e_x_y > 1e-10]
    assert e_x_y.shape[0] == 0

    e_y_y = error["y_y"].abs()
    e_y_y = e_y_y[e_y_y > 1e-10]
    assert e_y_y.shape[0] == 0

    count_pd = groups_pd.count()
    count_pd["name-string"] = count_pd.index
    count_pd = count_pd.reset_index(drop=True)

    new_fea_pd = df_expect_pd.merge(count_pd, on=["name-string"], how="left")
    new_fea_pd = new_fea_pd[["x_y"]]

    error = new_fea_count_host.subtract(new_fea_pd, axis=1)
    e_x_y = error["x_y"].abs()
    e_x_y = e_x_y[e_x_y > 0.0]
    assert e_x_y.shape[0] == 0


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("gpu_mem_trans_use", [0.00000001, 0.5])
def test_groupby_fit_merge_countonly(datasets, batch, dskey, gpu_mem_trans_use):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")

    grouby_stats = groupby.GroupByMomentsCal(
        col="name-string",
        col_count="x",
        cont_col=None,
        stats=["count"],
        gpu_mem_trans_use=gpu_mem_trans_use,
    )

    grouby_stats.fit(df_expect[["name-string", "x", "y"]])
    grouby_stats.fit_finalize()
    new_fea = grouby_stats.merge(df_expect[["name-string", "x", "y"]])
    new_fea_host = new_fea.to_pandas()
    new_fea_count_host = new_fea_host[["name-string_count"]]

    df_expect_pd = df_expect.to_pandas()
    groups_pd = df_expect_pd[["name-string", "x", "y"]].groupby(["name-string"])

    count_pd = groups_pd.count()
    count_pd["name-string"] = count_pd.index
    count_pd = count_pd.reset_index(drop=True)

    new_fea_pd = df_expect_pd.merge(count_pd, on=["name-string"], how="left")
    new_fea_pd = new_fea_pd[["x_y"]]

    error = new_fea_count_host.subtract(new_fea_pd, axis=1)
    e_x_y = error["x_y"].abs()
    e_x_y = e_x_y[e_x_y > 0.0]
    assert e_x_y.shape[0] == 0


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("gpu_mem_trans_use", [0.00000001, 0.5])
def test_groupby_fit_merge_mean(datasets, batch, dskey, gpu_mem_trans_use):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")

    grouby_stats = groupby.GroupByMomentsCal(
        col="name-string",
        col_count="x",
        cont_col=["x", "y"],
        stats=["mean"],
        gpu_mem_trans_use=gpu_mem_trans_use,
    )

    grouby_stats.fit(df_expect[["name-string", "x", "y"]])
    grouby_stats.fit_finalize()
    new_fea = grouby_stats.merge(df_expect[["name-string", "x", "y"]])
    new_fea_host = new_fea.to_pandas()
    new_fea_mean_host = new_fea_host[["name-string_x_mean", "name-string_y_mean"]]

    df_expect_pd = df_expect.to_pandas()
    groups_pd = df_expect_pd[["name-string", "x", "y"]].groupby(["name-string"])

    mean_pd = groups_pd.mean()
    mean_pd["name-string"] = mean_pd.index
    mean_pd = mean_pd.reset_index(drop=True)

    new_fea_pd = df_expect_pd.merge(mean_pd, on=["name-string"], how="left")
    new_fea_pd = new_fea_pd[["x_y", "y_y"]]

    error = new_fea_mean_host.subtract(new_fea_pd, axis=1)

    e_x_y = error["x_y"].abs()
    e_x_y = e_x_y[e_x_y > 1e-10]
    assert e_x_y.shape[0] == 0

    e_y_y = error["y_y"].abs()
    e_y_y = e_y_y[e_y_y > 1e-10]
    assert e_y_y.shape[0] == 0


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("gpu_mem_trans_use", [0.00000001, 0.5])
def test_groupby_fit_merge_stdvar(datasets, batch, dskey, gpu_mem_trans_use):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")

    grouby_stats = groupby.GroupByMomentsCal(
        col="name-string",
        col_count="x",
        cont_col=["x", "y"],
        stats=["std", "var"],
        gpu_mem_trans_use=gpu_mem_trans_use,
    )

    grouby_stats.fit(df_expect[["name-string", "x", "y"]])
    grouby_stats.fit_finalize()
    new_fea = grouby_stats.merge(df_expect[["name-string", "x", "y"]])
    new_fea_host = new_fea.to_pandas()
    new_fea_std_host = new_fea_host[["name-string_x_std", "name-string_y_std"]]
    new_fea_var_host = new_fea_host[["name-string_x_var", "name-string_y_var"]]

    df_expect_pd = df_expect.to_pandas()
    groups_pd = df_expect_pd[["name-string", "x", "y"]].groupby(["name-string"])

    # STD test
    std_pd = groups_pd.std(ddof=1)
    std_pd["name-string"] = std_pd.index
    std_pd = std_pd.reset_index(drop=True)

    new_fea_pd = df_expect_pd.merge(std_pd, on=["name-string"], how="left")
    new_fea_pd = new_fea_pd[["x_y", "y_y"]]

    error = new_fea_std_host.subtract(new_fea_pd, axis=1)

    e_x_y = error["x_y"].abs()
    e_x_y = e_x_y[e_x_y > 1e-10]
    assert e_x_y.shape[0] == 0

    e_y_y = error["y_y"].abs()
    e_y_y = e_y_y[e_y_y > 1e-10]
    assert e_y_y.shape[0] == 0

    # VAR Test
    var_pd = groups_pd.var(ddof=1)
    var_pd["name-string"] = var_pd.index
    var_pd = var_pd.reset_index(drop=True)

    new_fea_pd = df_expect_pd.merge(var_pd, on=["name-string"], how="left")
    new_fea_pd = new_fea_pd[["x_y", "y_y"]]

    error = new_fea_var_host.subtract(new_fea_pd, axis=1)

    e_x_y = error["x_y"].abs()
    e_x_y = e_x_y[e_x_y > 1e-10]
    assert e_x_y.shape[0] == 0

    e_y_y = error["y_y"].abs()
    e_y_y = e_y_y[e_y_y > 1e-10]
    assert e_y_y.shape[0] == 0
