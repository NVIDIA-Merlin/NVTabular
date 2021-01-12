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
import math
import string

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from cudf.tests.utils import assert_eq
from dask.dataframe import assert_eq as assert_eq_dd
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
import nvtabular.io
from nvtabular import ColumnGroup
from nvtabular import ops as ops
from tests.conftest import mycols_csv, mycols_pq


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
# TODO: dask workflow doesn't support min/max on string columns, so won't work
# with op_columns=None
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
def test_normalize_minmax(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cont_features = op_columns >> ops.NormalizeMinMax()
    processor = nvtabular.Workflow(cont_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    new_gdf.index = df.index  # Make sure index is aligned for checks
    for col in op_columns:
        col_min = df[col].min()
        assert col_min == pytest.approx(processor.column_group.op.mins[col], 1e-2)
        col_max = df[col].max()
        assert col_max == pytest.approx(processor.column_group.op.maxs[col], 1e-2)
        df[col] = (df[col] - processor.column_group.op.mins[col]) / (
            processor.column_group.op.maxs[col] - processor.column_group.op.mins[col]
        )
        assert np.all((df[col] - new_gdf[col]).abs().values <= 1e-2)


@pytest.mark.parametrize("cat_groups", ["Author", [["Author", "Engaging-User"]]])
@pytest.mark.parametrize("kfold", [1, 3])
@pytest.mark.parametrize("fold_seed", [None, 42])
def test_target_encode(tmpdir, cat_groups, kfold, fold_seed):
    df = cudf.DataFrame(
        {
            "Author": list(string.ascii_uppercase),
            "Engaging-User": list(string.ascii_lowercase),
            "Cost": range(26),
            "Post": [0, 1] * 13,
        }
    )
    df = dask_cudf.from_cudf(df, npartitions=3)

    cont_names = ["Cost"]
    te_features = cat_groups >> ops.TargetEncoding(
        cont_names,
        out_path=str(tmpdir),
        kfold=kfold,
        out_dtype="float32",
        fold_seed=fold_seed,
        drop_folds=False,  # Keep folds to validate
    )

    cont_features = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()
    workflow = nvt.Workflow(te_features + cont_features + ["Author", "Engaging-User"])
    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

    if kfold > 1:
        # Cat columns are unique.
        # Make sure __fold__ mapping is correct
        if cat_groups == "Author":
            name = "__fold___Author"
            cols = ["__fold__", "Author"]
        else:
            name = "__fold___Author_Engaging-User"
            cols = ["__fold__", "Author", "Engaging-User"]
        check = cudf.io.read_parquet(te_features.op.stats[name])
        check = check[cols].sort_values(cols).reset_index(drop=True)
        df_out_check = df_out[cols].sort_values(cols).reset_index(drop=True)
        assert_eq(check, df_out_check)


@pytest.mark.parametrize("npartitions", [1, 2])
def test_target_encode_multi(tmpdir, npartitions):

    cat_1 = np.asarray(["baaaa"] * 12)
    cat_2 = np.asarray(["baaaa"] * 6 + ["bbaaa"] * 3 + ["bcaaa"] * 3)
    num_1 = np.asarray([1, 1, 2, 2, 2, 1, 1, 5, 4, 4, 4, 4])
    num_2 = np.asarray([1, 1, 2, 2, 2, 1, 1, 5, 4, 4, 4, 4]) * 2
    df = cudf.DataFrame({"cat": cat_1, "cat2": cat_2, "num": num_1, "num_2": num_2})
    df = dask_cudf.from_cudf(df, npartitions=npartitions)

    cat_groups = ["cat", "cat2", ["cat", "cat2"]]
    te_features = cat_groups >> ops.TargetEncoding(
        ["num", "num_2"], out_path=str(tmpdir), kfold=1, p_smooth=5, out_dtype="float32"
    )

    workflow = nvt.Workflow(te_features)

    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

    assert "TE_cat_cat2_num" in df_out.columns
    assert "TE_cat_num" in df_out.columns
    assert "TE_cat2_num" in df_out.columns
    assert "TE_cat_cat2_num_2" in df_out.columns
    assert "TE_cat_num_2" in df_out.columns
    assert "TE_cat2_num_2" in df_out.columns

    assert_eq(df_out["TE_cat2_num"].values, df_out["TE_cat_cat2_num"].values)
    assert_eq(df_out["TE_cat2_num_2"].values, df_out["TE_cat_cat2_num_2"].values)
    assert df_out["TE_cat_num"].iloc[0] != df_out["TE_cat2_num"].iloc[0]
    assert df_out["TE_cat_num_2"].iloc[0] != df_out["TE_cat2_num_2"].iloc[0]
    assert math.isclose(df_out["TE_cat_num"].iloc[0], num_1.mean(), abs_tol=1e-4)
    assert math.isclose(df_out["TE_cat_num_2"].iloc[0], num_2.mean(), abs_tol=1e-3)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
def test_fill_median(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cont_features = op_columns >> nvt.ops.FillMedian()
    processor = nvt.Workflow(cont_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    new_gdf.index = df.index  # Make sure index is aligned for checks
    for col in op_columns:
        col_median = df[col].dropna().quantile(0.5, interpolation="linear")
        assert math.isclose(col_median, processor.column_group.op.medians[col], rel_tol=1e1)
        assert np.all((df[col].fillna(col_median) - new_gdf[col]).abs().values <= 1e-2)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
def test_log(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cont_features = op_columns >> nvt.ops.LogOp()
    processor = nvt.Workflow(cont_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    new_gdf.index = df.index  # Make sure index is aligned for checks
    assert new_gdf[op_columns] == np.log(df[op_columns].astype(np.float32))


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["name-string"], None])
def test_hash_bucket(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-string"]

    if op_columns is None:
        num_buckets = 10
    else:
        num_buckets = {column: 10 for column in op_columns}

    hash_features = cat_names >> ops.HashBucket(num_buckets)
    processor = nvt.Workflow(hash_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    # check sums for determinancy
    assert np.all(new_gdf[cat_names].values >= 0)
    assert np.all(new_gdf[cat_names].values <= 9)
    checksum = new_gdf[cat_names].sum().values
    new_gdf = processor.transform(dataset).to_ddf().compute()
    np.all(new_gdf[cat_names].sum().values == checksum)


def test_hash_bucket_lists(tmpdir):
    df = cudf.DataFrame(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Authors"]  # , "Engaging User"]

    dataset = nvt.Dataset(df)
    hash_features = cat_names >> ops.HashBucket(num_buckets=10)
    processor = nvt.Workflow(hash_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    # check to make sure that the same strings are hashed the same
    authors = new_gdf["Authors"].to_arrow().to_pylist()
    assert authors[0][0] == authors[1][0]  # 'User_A'
    assert authors[2][1] == authors[3][0]  # 'User_C'

    assert nvt.ops.get_embedding_sizes(processor)["Authors"][0] == 10


@pytest.mark.parametrize("engine", ["parquet"])
def test_fill_missing(tmpdir, df, dataset, engine):
    cont_names = ["x", "y"]
    cont_features = cont_names >> nvt.ops.FillMissing(fill_val=42)

    for col in cont_names:
        idx = np.random.choice(df.shape[0] - 1, int(df.shape[0] * 0.2))
        df[col].iloc[idx] = None

    df = df.reset_index()
    dataset = nvt.Dataset(df)
    processor = nvt.Workflow(cont_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    for col in cont_names:
        assert np.all((df[col].fillna(42) - new_gdf[col]).abs().values <= 1e-2)
        assert new_gdf[col].isna().sum() == 0


@pytest.mark.parametrize("engine", ["parquet"])
def test_dropna(tmpdir, df, dataset, engine):
    columns = mycols_pq if engine == "parquet" else mycols_csv
    dropna_features = columns >> ops.Dropna()

    processor = nvt.Workflow(dropna_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert new_gdf.columns.all() == df.columns.all()
    assert new_gdf.isnull().all().sum() < 1, "null values exist"


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], ["x", "y"]])
def test_normalize(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cont_features = op_columns >> ops.Normalize()
    processor = nvtabular.Workflow(cont_features)
    processor.fit(dataset)

    new_gdf = processor.transform(dataset).to_ddf().compute()
    new_gdf.index = df.index  # Make sure index is aligned for checks
    for col in op_columns:
        assert math.isclose(df[col].mean(), processor.column_group.op.means[col], rel_tol=1e-4)
        assert math.isclose(df[col].std(), processor.column_group.op.stds[col], rel_tol=1e-4)
        df[col] = (df[col] - processor.column_group.op.means[col]) / processor.column_group.op.stds[
            col
        ]
        assert np.all((df[col] - new_gdf[col]).abs().values <= 1e-2)


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("op_columns", [["x"]])
def test_normalize_upcastfloat64(tmpdir, dataset, gpu_memory_frac, engine, op_columns):
    df = cudf.DataFrame(
        {"x": [1.9e10, 2.3e16, 3.4e18, 1.6e19], "label": [1, 0, 1, 0]}, dtype="float32"
    )

    cont_features = op_columns >> ops.Normalize()
    processor = nvtabular.Workflow(cont_features)
    dataset = nvt.Dataset(df)
    processor.fit(dataset)

    new_gdf = processor.transform(dataset).to_ddf().compute()

    for col in op_columns:
        assert math.isclose(df[col].mean(), processor.column_group.op.means[col], rel_tol=1e-4)
        assert math.isclose(df[col].std(), processor.column_group.op.stds[col], rel_tol=1e-4)
        df[col] = (df[col] - processor.column_group.op.means[col]) / processor.column_group.op.stds[
            col
        ]
        assert np.all((df[col] - new_gdf[col]).abs().values <= 1e-2)


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
def test_lambdaop(tmpdir, df, dataset, gpu_memory_frac, engine):
    df_copy = df.copy()

    # Substring
    # Replacement
    substring = ColumnGroup(["name-cat", "name-string"]) >> (lambda col: col.str.slice(1, 3))
    processor = nvtabular.Workflow(substring)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"].str.slice(1, 3), check_index=False)
    assert_eq_dd(new_gdf["name-string"], df_copy["name-string"].str.slice(1, 3), check_index=False)

    # No Replacement from old API (skipped for other examples)
    substring = (
        ColumnGroup(["name-cat", "name-string"])
        >> (lambda col: col.str.slice(1, 3))
        >> ops.Rename(postfix="_slice")
    )
    processor = nvtabular.Workflow(["name-cat", "name-string"] + substring)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert_eq_dd(
        new_gdf["name-cat_slice"],
        df_copy["name-cat"].str.slice(1, 3),
        check_index=False,
        check_names=False,
    )
    assert_eq_dd(
        new_gdf["name-string_slice"],
        df_copy["name-string"].str.slice(1, 3),
        check_index=False,
        check_names=False,
    )
    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"], check_index=False)
    assert_eq_dd(new_gdf["name-string"], df_copy["name-string"], check_index=False)

    # Replace
    # Replacement
    oplambda = ColumnGroup(["name-cat", "name-string"]) >> (lambda col: col.str.replace("e", "XX"))
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert_eq_dd(new_gdf["name-cat"], df_copy["name-cat"].str.replace("e", "XX"), check_index=False)
    assert_eq_dd(
        new_gdf["name-string"], df_copy["name-string"].str.replace("e", "XX"), check_index=False
    )

    # astype
    # Replacement
    oplambda = ColumnGroup(["id"]) >> (lambda col: col.astype(float))
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert new_gdf["id"].dtype == "float64"

    # Workflow
    # Replacement
    oplambda = (
        ColumnGroup(["name-cat"])
        >> (lambda col: col.astype(str).str.slice(0, 1))
        >> ops.Categorify()
    )
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert is_integer_dtype(new_gdf["name-cat"].dtype)

    oplambda = (
        ColumnGroup(["name-cat", "name-string"]) >> ops.Categorify() >> (lambda col: col + 100)
    )
    processor = nvtabular.Workflow(oplambda)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert is_integer_dtype(new_gdf["name-cat"].dtype)
    assert np.sum(new_gdf["name-cat"] < 100) == 0


@pytest.mark.parametrize("freq_threshold", [0, 1, 2])
def test_categorify_lists(tmpdir, freq_threshold):
    df = cudf.DataFrame(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Authors", "Engaging User"]
    label_name = ["Post"]

    cat_features = cat_names >> ops.Categorify(out_path=str(tmpdir), freq_threshold=freq_threshold)

    workflow = nvt.Workflow(cat_features + label_name)
    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute()

    # Columns are encoded independently
    if freq_threshold < 2:
        assert df_out["Authors"].to_arrow().to_pylist() == [[1], [1, 4], [2, 3], [3]]
    else:
        assert df_out["Authors"].to_arrow().to_pylist() == [[1], [1, 0], [0, 2], [2]]


@pytest.mark.parametrize("cat_names", [[["Author", "Engaging User"]], ["Author", "Engaging User"]])
@pytest.mark.parametrize("kind", ["joint", "combo"])
def test_categorify_multi(tmpdir, cat_names, kind):
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )

    label_name = ["Post"]

    cats = cat_names >> ops.Categorify(out_path=str(tmpdir), encode_type=kind)

    workflow = nvt.Workflow(cats + label_name)

    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

    if len(cat_names) == 1:
        if kind == "joint":
            # Columns are encoded jointly
            assert df_out["Author"].to_arrow().to_pylist() == [1, 5, 2, 3]
            assert df_out["Engaging User"].to_arrow().to_pylist() == [2, 2, 1, 4]
        else:
            # Column combinations are encoded
            assert df_out["Author_Engaging User"].to_arrow().to_pylist() == [1, 4, 2, 3]
    else:
        # Columns are encoded independently
        assert df_out["Author"].to_arrow().to_pylist() == [1, 4, 2, 3]
        assert df_out["Engaging User"].to_arrow().to_pylist() == [2, 2, 1, 3]


def test_categorify_multi_combo(tmpdir):
    cat_names = [["Author", "Engaging User"], ["Author"], "Engaging User"]
    kind = "combo"
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )

    label_name = ["Post"]
    cats = cat_names >> ops.Categorify(out_path=str(tmpdir), encode_type=kind)
    workflow = nvt.Workflow(cats + label_name)
    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

    # Column combinations are encoded
    assert df_out["Author"].to_arrow().to_pylist() == [1, 4, 2, 3]
    assert df_out["Engaging User"].to_arrow().to_pylist() == [2, 2, 1, 3]
    assert df_out["Author_Engaging User"].to_arrow().to_pylist() == [1, 4, 2, 3]


@pytest.mark.parametrize("freq_limit", [None, 0, {"Author": 3, "Engaging User": 4}])
@pytest.mark.parametrize("buckets", [None, 10, {"Author": 10, "Engaging User": 20}])
@pytest.mark.parametrize("search_sort", [True, False])
def test_categorify_freq_limit(tmpdir, freq_limit, buckets, search_sort):
    df = cudf.DataFrame(
        {
            "Author": [
                "User_A",
                "User_E",
                "User_B",
                "User_C",
                "User_A",
                "User_E",
                "User_B",
                "User_C",
                "User_B",
                "User_C",
            ],
            "Engaging User": [
                "User_B",
                "User_B",
                "User_A",
                "User_D",
                "User_B",
                "User_c",
                "User_A",
                "User_D",
                "User_D",
                "User_D",
            ],
        }
    )

    isfreqthr = (isinstance(freq_limit, int) and freq_limit > 0) or (isinstance(freq_limit, dict))

    if (not search_sort and isfreqthr) or (search_sort and not isfreqthr):
        cat_names = ["Author", "Engaging User"]

        cats = cat_names >> ops.Categorify(
            freq_threshold=freq_limit,
            out_path=str(tmpdir),
            search_sorted=search_sort,
            num_buckets=buckets,
        )

        workflow = nvt.Workflow(cats)
        df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute(scheduler="synchronous")

        if freq_limit and not buckets:
            # Column combinations are encoded
            if isinstance(freq_limit, dict):
                assert df_out["Author"].max() == 2
                assert df_out["Engaging User"].max() == 1
            else:
                assert len(df["Author"].unique()) == df_out["Author"].max()
                assert len(df["Engaging User"].unique()) == df_out["Engaging User"].max()
        elif not freq_limit and buckets:
            if isinstance(buckets, dict):
                assert df_out["Author"].max() <= 9
                assert df_out["Engaging User"].max() <= 19
            else:
                assert df_out["Author"].max() <= 9
                assert df_out["Engaging User"].max() <= 9
        elif freq_limit and buckets:
            if isinstance(buckets, dict) and isinstance(buckets, dict):
                assert (
                    df_out["Author"].max()
                    <= (df["Author"].hash_values() % buckets["Author"]).max() + 2 + 1
                )
                assert (
                    df_out["Engaging User"].max()
                    <= (df["Engaging User"].hash_values() % buckets["Engaging User"]).max() + 1 + 1
                )


@pytest.mark.parametrize("groups", [[["Author", "Engaging-User"]], "Author"])
def test_joingroupby_multi(tmpdir, groups):

    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_A", "User_A", "User_B"],
            "Engaging-User": ["User_B", "User_B", "User_C", "User_C"],
            "Cost": [100.0, 200.0, 300.0, 400.0],
            "Post": [1, 2, 3, 4],
        }
    )

    groupby_features = groups >> ops.JoinGroupby(
        out_path=str(tmpdir), stats=["sum"], cont_names=["Cost"]
    )
    workflow = nvt.Workflow(groupby_features + "Post")

    df_out = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute()

    if isinstance(groups, list):
        # Join on ["Author", "Engaging-User"]
        assert df_out["Author_Engaging-User_Cost_sum"].to_arrow().to_pylist() == [
            300.0,
            300.0,
            300.0,
            400.0,
        ]
    else:
        # Join on ["Author"]
        assert df_out["Author_Cost_sum"].to_arrow().to_pylist() == [600.0, 600.0, 600.0, 400.0]


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("kind_ext", ["cudf", "pandas", "arrow", "parquet", "csv"])
@pytest.mark.parametrize("cache", ["host", "device"])
@pytest.mark.parametrize("how", ["left", "inner"])
@pytest.mark.parametrize("drop_duplicates", [True, False])
def test_join_external(tmpdir, df, dataset, engine, kind_ext, cache, how, drop_duplicates):

    # Define "external" table
    shift = 100
    df_ext = df[["id"]].copy().sort_values("id")
    df_ext["new_col"] = df_ext["id"] + shift
    df_ext["new_col_2"] = "keep"
    df_ext["new_col_3"] = "ignore"
    df_ext_check = df_ext.copy()
    if kind_ext == "pandas":
        df_ext = df_ext.to_pandas()
    elif kind_ext == "arrow":
        df_ext = df_ext.to_arrow()
    elif kind_ext == "parquet":
        path = tmpdir.join("external.parquet")
        df_ext.to_parquet(path)
        df_ext = path
    elif kind_ext == "csv":
        path = tmpdir.join("external.csv")
        df_ext.to_csv(path)
        df_ext = path

    # Define Op
    on = "id"
    columns_left = list(df.columns)
    columns_ext = ["id", "new_col", "new_col_2"]
    df_ext_check = df_ext_check[columns_ext]
    if drop_duplicates:
        df_ext_check.drop_duplicates(ignore_index=True, inplace=True)
    joined = nvt.ColumnGroup(columns_left) >> nvt.ops.JoinExternal(
        df_ext,
        on,
        how=how,
        columns_ext=columns_ext,
        cache=cache,
        drop_duplicates_ext=drop_duplicates,
    )

    gdf = df.reset_index()
    dataset = nvt.Dataset(gdf)
    processor = nvt.Workflow(joined)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute().reset_index()

    check_gdf = gdf.merge(df_ext_check, how=how, on=on)
    assert len(check_gdf) == len(new_gdf)
    assert (new_gdf["id"] + shift).all() == new_gdf["new_col"].all()
    assert gdf["id"].all() == new_gdf["id"].all()
    assert "new_col_2" in new_gdf.columns
    assert "new_col_3" not in new_gdf.columns


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
def test_filter(tmpdir, df, dataset, gpu_memory_frac, engine):
    cont_names = ["x", "y"]
    filtered = cont_names >> ops.Filter(f=lambda df: df[df["y"] > 0.5])
    processor = nvtabular.Workflow(filtered)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute().reset_index()
    filter_df = df[df["y"] > 0.5].reset_index()
    for col in cont_names:
        assert np.all((new_gdf[col] - filter_df[col]).abs().values <= 1e-2)

    # return isnull() rows
    for col in cont_names:
        idx = np.random.choice(df.shape[0] - 1, int(df.shape[0] * 0.2))
        df[col].iloc[idx] = None

    dataset = nvt.Dataset(df)
    filtered = cont_names >> ops.Filter(f=lambda df: df[df.x.isnull()])
    processor = nvtabular.Workflow(filtered)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert new_gdf.shape[0] < df.shape[0], "null values do not exist"

    # again testing filtering by returning a series rather than a df
    filtered = cont_names >> ops.Filter(f=lambda df: df.x.isnull())
    processor = nvtabular.Workflow(filtered)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert new_gdf.shape[0] < df.shape[0], "null values do not exist"

    # if the filter returns an invalid type we should get an exception immediately
    # (rather than causing problems downstream in the workflow)
    filtered = cont_names >> ops.Filter(f=lambda df: "some invalid value")
    processor = nvtabular.Workflow(filtered)
    with pytest.raises(ValueError):
        new_gdf = processor.transform(dataset).to_ddf().compute()


def test_difference_lag():
    df = cudf.DataFrame(
        {"userid": [0, 0, 0, 1, 1, 2], "timestamp": [1000, 1005, 1100, 2000, 2001, 3000]}
    )

    diff_features = ["timestamp"] >> ops.DifferenceLag(partition_cols=["userid"], shift=[1, -1])
    dataset = nvt.Dataset(df)
    processor = nvtabular.Workflow(diff_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    assert new_gdf["timestamp_difference_lag_1"][0] is cudf.NA
    assert new_gdf["timestamp_difference_lag_1"][1] == 5
    assert new_gdf["timestamp_difference_lag_1"][2] == 95
    assert new_gdf["timestamp_difference_lag_1"][3] is cudf.NA

    assert new_gdf["timestamp_difference_lag_-1"][0] == -5
    assert new_gdf["timestamp_difference_lag_-1"][1] == -95
    assert new_gdf["timestamp_difference_lag_-1"][2] is cudf.NA
    assert new_gdf["timestamp_difference_lag_-1"][3] == -1
    assert new_gdf["timestamp_difference_lag_-1"][5] is cudf.NA


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
def test_hashed_cross(tmpdir, df, dataset, gpu_memory_frac, engine):
    # TODO: add tests for > 2 features, multiple crosses, etc.
    cat_names = [["name-string", "id"]]
    num_buckets = 10

    hashed_cross = cat_names >> ops.HashedCross(num_buckets)
    dataset = nvt.Dataset(df)
    processor = nvtabular.Workflow(hashed_cross)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    # check sums for determinancy
    new_column_name = "_X_".join(cat_names[0])
    assert np.all(new_gdf[new_column_name].values >= 0)
    assert np.all(new_gdf[new_column_name].values <= 9)
    checksum = new_gdf[new_column_name].sum()
    new_gdf = processor.transform(dataset).to_ddf().compute()
    assert new_gdf[new_column_name].sum() == checksum


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
def test_bucketized(tmpdir, df, dataset, gpu_memory_frac, engine):
    cont_names = ["x", "y"]
    boundaries = [[-1, 0, 1], [-4, 100]]

    bucketize_op = ops.Bucketize({name: boundary for name, boundary in zip(cont_names, boundaries)})

    bucket_features = cont_names >> bucketize_op
    processor = nvtabular.Workflow(bucket_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    for col, bs in zip(cont_names, boundaries):
        assert np.all(new_gdf[col].values >= 0)
        assert np.all(new_gdf[col].values <= len(bs))
        # TODO: add checks for correctness here that don't just
        # repeat the existing logic
