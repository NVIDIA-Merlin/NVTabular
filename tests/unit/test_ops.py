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
import os
import string

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from cudf.tests.utils import assert_eq
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
import nvtabular.io
from nvtabular import ops as ops
from tests.conftest import get_cats, mycols_csv, mycols_pq


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
# TODO: dask workflow doesn't support min/max on string columns, so won't work
# with op_columns=None
@pytest.mark.parametrize("op_columns", [["x"]])
def test_minmax(tmpdir, client, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]

    config = nvtabular.workflow.get_new_config()
    config["PP"]["all"] = [ops.MinMax(columns=op_columns)]

    processor = nvtabular.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )
    processor.update_stats(dataset)
    x_min = df["x"].min()

    assert x_min == pytest.approx(processor.stats["mins"]["x"], 1e-2)
    x_max = df["x"].max()
    assert x_max == pytest.approx(processor.stats["maxs"]["x"], 1e-2)
    if not op_columns:
        name_min = min(df["name-string"].tolist())
        name_max = max(df["name-string"].tolist())
        assert name_min == processor.stats["mins"]["name-string"]
        y_max = df["y"].max()
        y_min = df["y"].min()
        assert y_max == processor.stats["maxs"]["y"]
        assert name_max == processor.stats["maxs"]["name-string"]
        assert y_min == processor.stats["mins"]["y"]


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
def test_moments(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.Moments(columns=op_columns)]

    processor = nvtabular.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )

    processor.update_stats(dataset)

    assert df.x.count() == processor.stats["counts"]["x"]
    assert df.x.count() == 4321

    # Check mean and std
    assert math.isclose(df.x.mean(), processor.stats["means"]["x"], rel_tol=1e-4)
    assert math.isclose(df.x.std(), processor.stats["stds"]["x"], rel_tol=1e-3)
    if not op_columns:
        assert math.isclose(df.y.mean(), processor.stats["means"]["y"], rel_tol=1e-4)
        assert math.isclose(df.id.mean(), processor.stats["means"]["id"], rel_tol=1e-4)

        assert math.isclose(df.y.std(), processor.stats["stds"]["y"], rel_tol=1e-3)
        assert math.isclose(df.id.std(), processor.stats["stds"]["id"], rel_tol=1e-3)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["name-string"], None])
def test_encoder(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    encoder = ops.GroupbyStatistics(columns=op_columns)
    config = nvt.workflow.get_new_config()
    config["PP"]["categorical"] = [encoder]

    processor = nvt.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )
    processor.update_stats(dataset)

    if engine == "parquet" and not op_columns:
        cats_expected0 = df["name-cat"].unique().values_host
        cats0 = get_cats(processor, "name-cat")
        assert cats0.tolist() == [None] + cats_expected0.tolist()

    cats_expected1 = df["name-string"].unique().values_host
    cats1 = get_cats(processor, "name-string")
    assert cats1.tolist() == [None] + cats_expected1.tolist()


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("groups", [[["name-cat", "name-string"], "name-cat"], "name-string"])
@pytest.mark.parametrize("concat_groups", [True, False])
def test_multicolumn_cats(tmpdir, df, dataset, engine, groups, concat_groups):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    encoder = ops.GroupbyStatistics(
        columns=groups,
        cont_names=None if concat_groups else ["x"],
        stats=None if concat_groups else ["count", "mean"],
        out_path=str(tmpdir),
        concat_groups=concat_groups,
    )
    config = nvt.workflow.get_new_config()
    config["PP"]["categorical"] = [encoder]

    processor = nvt.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )
    processor.update_stats(dataset)

    groups = [groups] if isinstance(groups, str) else groups
    for group in groups:
        group = [group] if isinstance(group, str) else group
        prefix = "unique." if concat_groups else "cat_stats."
        fn = prefix + "_".join(group) + ".parquet"
        cudf.read_parquet(os.path.join(tmpdir, "categories", fn))


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("groups", [[["name-cat", "name-string"]], "name-string"])
@pytest.mark.parametrize("kfold", [3])
def test_groupby_folds(tmpdir, df, dataset, engine, groups, kfold):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    gb_stats = ops.GroupbyStatistics(
        columns=None,
        out_path=str(tmpdir),
        kfold=kfold,
        fold_groups=groups,
        stats=["count", "sum"],
        cont_names=["y"],
    )
    config = nvt.workflow.get_new_config()
    config["PP"]["categorical"] = [gb_stats]

    processor = nvt.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )
    processor.update_stats(dataset)
    for group, path in processor.stats["categories"].items():
        df = cudf.read_parquet(path)
        assert "__fold__" in df.columns


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

    cat_names = ["Author", "Engaging-User"]
    cont_names = ["Cost"]
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)
    processor.add_feature([ops.FillMissing(), ops.Clip(min_value=0), ops.LogOp()])
    processor.add_preprocess(
        ops.TargetEncoding(
            cat_groups,
            "Cost",  # cont_target
            out_path=str(tmpdir),
            kfold=kfold,
            out_col="test_name",
            out_dtype="float32",
            fold_seed=fold_seed,
            drop_folds=False,  # Keep folds to validate
        )
    )
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

    assert "test_name" in df_out.columns
    assert df_out["test_name"].dtype == "float32"

    if kfold > 1:
        # Cat columns are unique.
        # Make sure __fold__ mapping is correct
        if cat_groups == "Author":
            name = "__fold___Author"
            cols = ["__fold__", "Author"]
        else:
            name = "__fold___Author_Engaging-User"
            cols = ["__fold__", "Author", "Engaging-User"]
        check = cudf.io.read_parquet(processor.stats["te_stats"][name])
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

    cat_names = ["cat", "cat2"]
    cont_names = ["num", "num_2"]
    label_name = []
    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    cat_groups = ["cat", "cat2", ["cat", "cat2"]]

    processor.add_preprocess(
        ops.TargetEncoding(
            cat_groups,
            ["num", "num_2"],  # cont_target
            out_path=str(tmpdir),
            kfold=1,
            p_smooth=5,
            out_dtype="float32",
        )
    )
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

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
@pytest.mark.parametrize("op_columns", [["x"], None])
def test_median(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.Median(columns=op_columns)]

    processor = nvt.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )

    processor.update_stats(dataset)

    # Check median (TODO: Improve the accuracy)
    x_median = df.x.dropna().quantile(0.5, interpolation="linear")
    assert math.isclose(x_median, processor.stats["medians"]["x"], rel_tol=1e1)
    if not op_columns:
        y_median = df.y.dropna().quantile(0.5, interpolation="linear")
        id_median = df.id.dropna().quantile(0.5, interpolation="linear")
        assert math.isclose(y_median, processor.stats["medians"]["y"], rel_tol=1e1)
        assert math.isclose(id_median, processor.stats["medians"]["id"], rel_tol=1e1)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
def test_log(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cont_names = ["x", "y", "id"]
    log_op = ops.LogOp(columns=op_columns)

    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = cont_names

    for gdf in dataset.to_iter():
        new_gdf = log_op.apply_op(gdf, columns_ctx, "continuous")
        assert new_gdf[cont_names] == np.log(gdf[cont_names].astype(np.float32))


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["name-string"], None])
def test_hash_bucket(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-string"]

    if op_columns is None:
        num_buckets = 10
    else:
        num_buckets = {column: 10 for column in op_columns}
    hash_bucket_op = ops.HashBucket(num_buckets)

    columns_ctx = {}
    columns_ctx["categorical"] = {}
    columns_ctx["categorical"]["base"] = cat_names

    # check sums for determinancy
    checksums = []
    for gdf in dataset.to_iter():
        new_gdf = hash_bucket_op.apply_op(gdf, columns_ctx, "categorical")
        assert np.all(new_gdf[cat_names].values >= 0)
        assert np.all(new_gdf[cat_names].values <= 9)
        checksums.append(new_gdf[cat_names].sum().values)

    for checksum, gdf in zip(checksums, dataset.to_iter()):
        new_gdf = hash_bucket_op.apply_op(gdf, columns_ctx, "categorical")
        assert np.all(new_gdf[cat_names].sum().values == checksum)


def test_hash_bucket_lists(tmpdir):
    df = cudf.DataFrame(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Authors"]  # , "Engaging User"]
    cont_names = []
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)
    processor.add_preprocess(ops.HashBucket(num_buckets=10))
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

    # check to make sure that the same strings are hashed the same
    authors = df_out["Authors"].to_arrow().to_pylist()
    assert authors[0][0] == authors[1][0]  # 'User_A'
    assert authors[2][1] == authors[3][0]  # 'User_C'


@pytest.mark.parametrize("engine", ["parquet"])
def test_fill_missing(tmpdir, df, dataset, engine):
    op = nvt.ops.FillMissing(42)

    cont_names = ["x", "y"]
    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = cont_names
    for col in cont_names:
        idx = np.random.choice(df.shape[0] - 1, int(df.shape[0] * 0.2))
        df[col].iloc[idx] = None

    transformed = cudf.concat([op.apply_op(df, columns_ctx, "continuous")])
    assert_eq(transformed[cont_names], df[cont_names].fillna(42))


@pytest.mark.parametrize("engine", ["parquet"])
def test_dropna(tmpdir, df, dataset, engine):
    dropna = ops.Dropna()
    columns = mycols_pq if engine == "parquet" else mycols_csv

    columns_ctx = {}
    columns_ctx["all"] = {}
    columns_ctx["all"]["base"] = columns

    for gdf in dataset.to_iter():
        new_gdf = dropna.apply_op(gdf, columns_ctx, "all")
        assert new_gdf.columns.all() == gdf.columns.all()
        assert new_gdf.isnull().all().sum() < 1, "null values exist"


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
def test_normalize(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.Moments(columns=op_columns)]

    processor = nvtabular.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )

    processor.update_stats(dataset)

    op = ops.Normalize()

    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = op_columns or cont_names

    new_gdf = op.apply_op(df, columns_ctx, "continuous", stats_context=processor.stats)
    df["x"] = (df["x"] - processor.stats["means"]["x"]) / processor.stats["stds"]["x"]
    assert new_gdf["x"].equals(df["x"])


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
def test_normalize_minmax(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.MinMax()]

    processor = nvtabular.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )

    processor.update_stats(dataset)

    op = ops.NormalizeMinMax()

    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = cont_names

    new_gdf = op.apply_op(df, columns_ctx, "continuous", stats_context=processor.stats)
    df["x"] = (df["x"] - processor.stats["mins"]["x"]) / (
        processor.stats["maxs"]["x"] - processor.stats["mins"]["x"]
    )
    assert new_gdf["x"].equals(df["x"])


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
def test_lambdaop(tmpdir, df, dataset, gpu_memory_frac, engine, client):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]
    columns = mycols_pq if engine == "parquet" else mycols_csv

    df_copy = df.copy()

    config = nvt.workflow.get_new_config()

    processor = nvtabular.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        client=client,
    )

    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = cont_names
    columns_ctx["all"] = {}
    columns_ctx["all"]["base"] = columns

    # Substring
    # Replacement
    op = ops.LambdaOp(
        op_name="slice",
        f=lambda col, gdf: col.str.slice(1, 3),
        columns=["name-cat", "name-string"],
        replace=True,
    )

    new_gdf = op.apply_op(df, columns_ctx, "all", stats_context=None)
    assert new_gdf["name-cat"].equals(df_copy["name-cat"].str.slice(1, 3))
    assert new_gdf["name-string"].equals(df_copy["name-string"].str.slice(1, 3))

    # No Replacement
    df = df_copy.copy()
    op = ops.LambdaOp(
        op_name="slice",
        f=lambda col, gdf: col.str.slice(1, 3),
        columns=["name-cat", "name-string"],
        replace=False,
    )
    new_gdf = op.apply_op(df, columns_ctx, "all", stats_context=None)
    assert new_gdf["name-cat_slice"].equals(df_copy["name-cat"].str.slice(1, 3))
    assert new_gdf["name-string_slice"].equals(df_copy["name-string"].str.slice(1, 3))
    assert new_gdf["name-cat"].equals(df_copy["name-cat"])
    assert new_gdf["name-string"].equals(df_copy["name-string"])

    # Replace
    # Replacement
    df = df_copy.copy()
    op = ops.LambdaOp(
        op_name="replace",
        f=lambda col, gdf: col.str.replace("e", "XX"),
        columns=["name-cat", "name-string"],
        replace=True,
    )

    new_gdf = op.apply_op(df, columns_ctx, "all", stats_context=None)
    assert new_gdf["name-cat"].equals(df_copy["name-cat"].str.replace("e", "XX"))
    assert new_gdf["name-string"].equals(df_copy["name-string"].str.replace("e", "XX"))

    # No Replacement
    df = df_copy.copy()
    op = ops.LambdaOp(
        op_name="replace",
        f=lambda col, gdf: col.str.replace("e", "XX"),
        columns=["name-cat", "name-string"],
        replace=False,
    )
    new_gdf = op.apply_op(df, columns_ctx, "all", stats_context=None)
    assert new_gdf["name-cat_replace"].equals(df_copy["name-cat"].str.replace("e", "XX"))
    assert new_gdf["name-string_replace"].equals(df_copy["name-string"].str.replace("e", "XX"))
    assert new_gdf["name-cat"].equals(df_copy["name-cat"])
    assert new_gdf["name-string"].equals(df_copy["name-string"])

    # astype
    # Replacement
    df = df_copy.copy()
    op = ops.LambdaOp(
        op_name="astype", f=lambda col, gdf: col.astype(float), columns=["id"], replace=True
    )
    new_gdf = op.apply_op(df, columns_ctx, "all", stats_context=None)
    assert new_gdf["id"].dtype == "float64"

    # Workflow
    # Replacement
    import glob

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(
        [
            ops.LambdaOp(
                op_name="slice",
                f=lambda col, gdf: col.astype(str).str.slice(0, 1),
                columns=["name-cat"],
                replace=True,
            ),
            ops.Categorify(),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out1")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
    )

    dataset_2 = nvtabular.io.Dataset(
        glob.glob(str(outdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac
    )
    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)
    assert is_integer_dtype(df_pp["name-cat"].dtype)

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(
        [
            ops.Categorify(),
            ops.LambdaOp(op_name="add100", f=lambda col, gdf: col + 100, replace=True),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out2")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
    )

    dataset_2 = nvtabular.io.Dataset(
        glob.glob(str(outdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac
    )
    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)
    assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert np.sum(df_pp["name-cat"] < 100) == 0

    # Workflow
    # No Replacement
    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(
        [
            ops.LambdaOp(
                op_name="slice",
                f=lambda col, gdf: col.astype(str).str.slice(0, 1),
                columns=["name-cat"],
                replace=False,
            ),
            ops.Categorify(),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out3")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
    )
    dataset_2 = nvtabular.io.Dataset(
        glob.glob(str(outdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac
    )
    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)

    assert df_pp["name-cat"].dtype == "O"
    print(df_pp)
    assert is_integer_dtype(df_pp["name-cat_slice"].dtype)
    assert np.sum(df_pp["name-cat_slice"] == 0) == 0

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(
        [
            ops.Categorify(),
            ops.LambdaOp(op_name="add100", f=lambda col, gdf: col + 100, replace=False),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out4")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
    )

    dataset_2 = nvtabular.io.Dataset(
        glob.glob(str(outdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac
    )
    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)
    assert is_integer_dtype(df_pp["name-cat_add100"].dtype)
    assert np.sum(df_pp["name-cat_add100"] < 100) == 0

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(
        [
            ops.LambdaOp(op_name="mul0", f=lambda col, gdf: col * 0, columns=["x"], replace=False),
            ops.LambdaOp(op_name="add100", f=lambda col, gdf: col + 100, replace=False),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out5")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
    )

    dataset_2 = nvtabular.io.Dataset(
        glob.glob(str(outdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac
    )
    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)
    assert np.sum(df_pp["x_mul0_add100"] < 100) == 0


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
    cont_names = []
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)
    processor.add_preprocess(ops.Categorify(out_path=str(tmpdir), freq_threshold=freq_threshold))
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

    # Columns are encoded independently
    if freq_threshold < 2:
        assert df_out["Authors"].to_arrow().to_pylist() == [[1], [1, 4], [2, 3], [3]]
    else:
        assert df_out["Authors"].to_arrow().to_pylist() == [[1], [1, 0], [0, 2], [2]]


@pytest.mark.parametrize("groups", [[["Author", "Engaging User"]], None])
@pytest.mark.parametrize("kind", ["joint", "combo"])
def test_categorify_multi(tmpdir, groups, kind):

    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )

    cat_names = ["Author", "Engaging User"]
    cont_names = []
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(ops.Categorify(columns=groups, out_path=str(tmpdir), encode_type=kind))
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

    if groups:
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
    groups = [["Author", "Engaging User"], ["Author"], "Engaging User"]
    kind = "combo"
    df = pd.DataFrame(
        {
            "Author": ["User_A", "User_E", "User_B", "User_C"],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )

    cat_names = ["Author", "Engaging User"]
    cont_names = []
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(ops.Categorify(columns=groups, out_path=str(tmpdir), encode_type=kind))
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

    # Column combinations are encoded
    assert df_out["Author"].to_arrow().to_pylist() == [1, 4, 2, 3]
    assert df_out["Engaging User"].to_arrow().to_pylist() == [2, 2, 1, 3]
    assert df_out["Author_Engaging User"].to_arrow().to_pylist() == [1, 4, 2, 3]


@pytest.mark.parametrize("freq_limit", [None, 0, {"Author": 3, "Engaging User": 4}])
@pytest.mark.parametrize("search_sort", [True, False])
def test_categorify_freq_limit(tmpdir, freq_limit, search_sort):
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
        cont_names = []
        label_name = []

        processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

        processor.add_preprocess(
            ops.Categorify(
                columns=cat_names,
                freq_threshold=freq_limit,
                out_path=str(tmpdir),
                search_sorted=search_sort,
            )
        )
        processor.finalize()
        processor.apply(nvt.Dataset(df), output_format=None)
        df_out = processor.get_ddf().compute(scheduler="synchronous")

        # Column combinations are encoded
        if isinstance(freq_limit, dict):
            assert df_out["Author"].max() == 2
            assert df_out["Engaging User"].max() == 1
        else:
            assert len(df["Author"].unique()) == df_out["Author"].max()
            assert len(df["Engaging User"].unique()) == df_out["Engaging User"].max()


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

    cat_names = ["Author", "Engaging-User"]
    cont_names = ["Cost"]
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_preprocess(
        ops.JoinGroupby(columns=groups, out_path=str(tmpdir), stats=["sum"], cont_names=["Cost"])
    )
    processor.finalize()
    processor.apply(nvt.Dataset(df), output_format=None)
    df_out = processor.get_ddf().compute(scheduler="synchronous")

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
    columns_ext = ["id", "new_col", "new_col_2"]
    df_ext_check = df_ext_check[columns_ext]
    if drop_duplicates:
        df_ext_check.drop_duplicates(ignore_index=True, inplace=True)
    merge_op = ops.JoinExternal(
        df_ext,
        on,
        how=how,
        columns_ext=columns_ext,
        cache=cache,
        drop_duplicates_ext=drop_duplicates,
    )
    columns = mycols_pq if engine == "parquet" else mycols_csv
    columns_ctx = {}
    columns_ctx["all"] = {}
    columns_ctx["all"]["base"] = columns

    # Iterate, apply op, and check result
    for gdf in dataset.to_iter():
        new_gdf = merge_op.apply_op(gdf, columns_ctx, "all")
        check_gdf = gdf.merge(df_ext_check, how=how, on=on)
        assert len(check_gdf) == len(new_gdf)
        assert (new_gdf["id"] + shift).all() == new_gdf["new_col"].all()
        assert gdf["id"].all() == new_gdf["id"].all()
        assert "new_col_2" in new_gdf.columns
        assert "new_col_3" not in new_gdf.columns


@pytest.mark.parametrize("gpu_memory_frac", [0.1])
@pytest.mark.parametrize("engine", ["parquet"])
def test_filter(tmpdir, df, dataset, gpu_memory_frac, engine, client):

    cont_names = ["x", "y"]

    columns = mycols_pq if engine == "parquet" else mycols_csv
    columns_ctx = {}
    columns_ctx["all"] = {}
    columns_ctx["all"]["base"] = columns

    filter_op = ops.Filter(f=lambda df: df[df["y"] > 0.5])
    new_gdf = filter_op.apply_op(df, columns_ctx, "all", target_cols=columns)
    assert new_gdf.columns.all() == df.columns.all()

    # return isnull() rows
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = cont_names

    for col in cont_names:
        idx = np.random.choice(df.shape[0] - 1, int(df.shape[0] * 0.2))
        df[col].iloc[idx] = None

    filter_op = ops.Filter(f=lambda df: df[df.x.isnull()])
    new_gdf = filter_op.apply_op(df, columns_ctx, "all", target_cols=columns)
    assert new_gdf.columns.all() == df.columns.all()
    assert new_gdf.shape[0] < df.shape[0], "null values do not exist"

    # again testing filtering by returning a series rather than a df
    filter_op = ops.Filter(f=lambda df: df.x.isnull())
    new_gdf = filter_op.apply_op(df, columns_ctx, "all", target_cols=columns)
    assert new_gdf.columns.all() == df.columns.all()
    assert new_gdf.shape[0] < df.shape[0], "null values do not exist"

    # if the filter returns an invalid type we should get an exception immediately
    # (rather than causing problems downstream in the workflow)
    filter_op = ops.Filter(f=lambda df: "some invalid value")
    with pytest.raises(ValueError):
        filter_op.apply_op(df, columns_ctx, "all", target_cols=columns)


def test_difference_lag():
    df = cudf.DataFrame(
        {"userid": [0, 0, 0, 1, 1, 2], "timestamp": [1000, 1005, 1100, 2000, 2001, 3000]}
    )

    columns = ["userid", "timestamp"]
    columns_ctx = {}
    columns_ctx["all"] = {}
    columns_ctx["all"]["base"] = columns

    op = ops.DifferenceLag("userid", shift=[1, -1], columns=["timestamp"])
    new_gdf = op.apply_op(df, columns_ctx, "all", target_cols=["timestamp"])

    assert new_gdf["timestamp_DifferenceLag_1"][0] is None
    assert new_gdf["timestamp_DifferenceLag_1"][1] == 5
    assert new_gdf["timestamp_DifferenceLag_1"][2] == 95
    assert new_gdf["timestamp_DifferenceLag_1"][3] is None

    assert new_gdf["timestamp_DifferenceLag_-1"][0] == -5
    assert new_gdf["timestamp_DifferenceLag_-1"][1] == -95
    assert new_gdf["timestamp_DifferenceLag_-1"][2] is None
    assert new_gdf["timestamp_DifferenceLag_-1"][3] == -1
    assert new_gdf["timestamp_DifferenceLag_-1"][5] is None


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("use_dict", [True, False])
def test_hashed_cross(tmpdir, df, dataset, gpu_memory_frac, engine, use_dict):
    # TODO: add tests for > 2 features, multiple crosses, etc.
    cat_names = ("name-string", "id")
    num_buckets = 10

    if use_dict:
        hashed_cross_op = ops.HashedCross({cat_names: num_buckets})
    else:
        hashed_cross_op = ops.HashedCross([cat_names], [num_buckets])

    columns_ctx = {}
    columns_ctx["categorical"] = {}
    columns_ctx["categorical"]["base"] = list(cat_names)

    # check sums for determinancy
    checksums = []
    for gdf in dataset.to_iter():
        new_gdf = hashed_cross_op.apply_op(gdf, columns_ctx, "categorical")
        new_column_name = "_X_".join(cat_names)
        assert np.all(new_gdf[new_column_name].values >= 0)
        assert np.all(new_gdf[new_column_name].values <= 9)
        checksums.append(new_gdf[new_column_name].sum())

    for checksum, gdf in zip(checksums, dataset.to_iter()):
        new_gdf = hashed_cross_op.apply_op(gdf, columns_ctx, "categorical")
        assert new_gdf[new_column_name].sum() == checksum


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("use_dict", [True, False])
def test_bucketized(tmpdir, df, dataset, gpu_memory_frac, engine, use_dict):
    cont_names = ["x", "y"]
    boundaries = [[-1, 0, 1], [-4, 100]]

    if use_dict:
        bucketize_op = ops.Bucketize(
            {name: boundary for name, boundary in zip(cont_names, boundaries)}
        )
    else:
        bucketize_op = ops.Bucketize(boundaries, cont_names)

    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = list(cont_names)
    for gdf in dataset.to_iter():
        new_gdf = bucketize_op.apply_op(gdf, columns_ctx, "continuous")
        for col, bs in zip(cont_names, boundaries):
            assert np.all(new_gdf[col].values >= 0)
            assert np.all(new_gdf[col].values <= len(bs))
            # TODO: add checks for correctness here that don't just
            # repeat the existing logic
