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

import cudf
import numpy as np
import pytest
from cudf.tests.utils import assert_eq
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
import nvtabular.io
import nvtabular.ops as ops
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

    encoder = ops.CategoryStatistics(columns=op_columns)
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
@pytest.mark.parametrize(
    "groups",
    [
        None,
        # [["name-cat", "name-string"], "name-cat"], None,
    ],
)
def test_multicolumn_cats(tmpdir, df, dataset, engine, groups):
    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    # encoder = ops.CategoryStatistics(columns=groups)
    encoder = ops.CategoryStatistics()
    config = nvt.workflow.get_new_config()
    config["PP"]["categorical"] = [encoder]

    processor = nvt.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )
    processor.update_stats(dataset)

    pass


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
    config["PP"]["continuous"] = [ops.Moments()]

    processor = nvtabular.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, config=config
    )

    processor.update_stats(dataset)

    op = ops.Normalize()

    columns_ctx = {}
    columns_ctx["continuous"] = {}
    columns_ctx["continuous"]["base"] = cont_names

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
        preprocessing=True,
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
        preprocessing=True,
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
        preprocessing=True,
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
        preprocessing=True,
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
        op_name="astype",
        f=lambda col, gdf: col.astype(float),
        columns=["id"],
        preprocessing=True,
        replace=True,
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
                preprocessing=True,
                replace=True,
            ),
            ops.Categorify(),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out1")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle="partial", apply_ops=True
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
            ops.LambdaOp(
                op_name="add100", f=lambda col, gdf: col + 100, preprocessing=True, replace=True
            ),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out2")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle="partial", apply_ops=True
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
                preprocessing=True,
                replace=False,
            ),
            ops.Categorify(),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out3")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle="partial", apply_ops=True
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
            ops.LambdaOp(
                op_name="add100", f=lambda col, gdf: col + 100, preprocessing=True, replace=False
            ),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out4")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle="partial", apply_ops=True
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
            ops.LambdaOp(
                op_name="mul0",
                f=lambda col, gdf: col * 0,
                columns=["x"],
                preprocessing=True,
                replace=False,
            ),
            ops.LambdaOp(
                op_name="add100", f=lambda col, gdf: col + 100, preprocessing=True, replace=False
            ),
        ]
    )
    processor.finalize()
    processor.update_stats(dataset)
    outdir = tmpdir.mkdir("out5")
    processor.write_to_dataset(
        outdir, dataset, out_files_per_proc=10, shuffle="partial", apply_ops=True
    )

    dataset_2 = nvtabular.io.Dataset(
        glob.glob(str(outdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac
    )
    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)
    assert np.sum(df_pp["x_mul0_add100"] < 100) == 0
