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

import nvtabular as nvt
import nvtabular.io
import nvtabular.ops as ops
from tests.conftest import get_cats, mycols_csv, mycols_pq


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
# TODO: dask workflow doesn't support min/max on string columns, so won't work
# with op_columns=None
@pytest.mark.parametrize("op_columns", [["x"]])
@pytest.mark.parametrize("use_client", [True, False])
def test_minmax(tmpdir, client, df, dataset, gpu_memory_frac, engine, op_columns, use_client):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]

    config = nvtabular.workflow.get_new_config()
    config["PP"]["all"] = [ops.MinMax(columns=op_columns)]

    processor = nvtabular.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        **{"client": client} if use_client else {}
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
def test_moments(client, tmpdir, df, dataset, gpu_memory_frac, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.Moments(columns=op_columns)]

    processor = nvtabular.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        client=client,
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
@pytest.mark.parametrize("use_client", [True, False])
def test_encoder(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, client, use_client):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    encoder = ops.Encoder(columns=op_columns)
    config = nvt.workflow.get_new_config()
    config["PP"]["categorical"] = [encoder]
    processor = nvt.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        **{"client": client} if use_client else {}
    )
    processor.update_stats(dataset)

    if engine == "parquet" and not op_columns:
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = get_cats(processor, "name-cat")
        assert cats0 == ["None"] + cats_expected0

    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = get_cats(processor, "name-string")
    assert cats1 == ["None"] + cats_expected1


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
@pytest.mark.parametrize("use_client", [True, False])
def test_median(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, client, use_client):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.Median(columns=op_columns)]

    processor = nvt.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        **{"client": client} if use_client else {}
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

    transformed = cudf.concat(
        [op.apply_op(df, columns_ctx, "continuous") for df in dataset.to_iter()]
    )
    assert_eq(transformed[cont_names], df[cont_names].dropna(42))


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
def test_normalize(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, client):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.Moments()]

    processor = nvtabular.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        client=client,
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
def test_normalize_minmax(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, client):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["PP"]["continuous"] = [ops.MinMax()]

    processor = nvtabular.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        client=client,
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

@pytest.mark.parametrize("engine", ["parquet"])
def test_seriesops(tmpdir, datasets, gpu_memory_frac, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[allcols_csv]
        df2 = cudf.read_parquet(paths[1])[allcols_csv]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[allcols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[allcols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        columns = allcols_csv
    else:
        columns = allcols_csv

    data_itr = nvtabular.io.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    columns_ctx = {}
    columns_ctx["all"] = {}
    columns_ctx["all"]["base"] = columns

    # Substring
    # Replacement
    op_seriesOp = ops.SeriesOps(
        names="name-string",
        names_new=None,
        op_names=["str", "slice"],
        start=1,
        stop=3,
        preprocessing=True,
        replace=True,
    )

    for gdf in data_itr:
        str_slice = gdf["name-string"].str.slice(1, 3)
        new_gdf = op_seriesOp.apply_op(gdf, columns_ctx, "all")
        assert np.sum(new_gdf["name-string"] != str_slice) == 0

    # No Replacement
    op_seriesOp = ops.SeriesOps(
        names="name-string",
        names_new="name-string-new",
        op_names=["str", "slice"],
        start=1,
        stop=3,
        replace=False,
    )

    for gdf in data_itr:
        new_gdf = op_seriesOp.apply_op(gdf, columns_ctx, "all")
        assert np.sum(new_gdf["name-string-new"] != gdf["name-string"].str.slice(1, 3)) == 0
        assert np.sum(new_gdf["name-string"] != gdf["name-string"]) == 0

    # Replace
    # Replacement
    op_seriesOp = ops.SeriesOps(
        names="name-string",
        names_new=None,
        op_names=["str", "replace"],
        pat="e",
        repl="XX",
        preprocessing=True,
        replace=True,
    )

    for gdf in data_itr:
        str_replace = gdf["name-string"].str.replace("e", "XX")
        new_gdf = op_seriesOp.apply_op(gdf, columns_ctx, "all")
        assert np.sum(new_gdf["name-string"] != str_replace) == 0

    # No Replacement
    op_seriesOp = ops.SeriesOps(
        names="name-string",
        names_new="name-string-new",
        op_names=["str", "replace"],
        pat="e",
        repl="XX",
        replace=False,
    )

    for gdf in data_itr:
        new_gdf = op_seriesOp.apply_op(gdf, columns_ctx, "all")
        assert np.sum(new_gdf["name-string-new"] != gdf["name-string"].str.replace("e", "XX")) == 0
        assert np.sum(new_gdf["name-string"] != gdf["name-string"]) == 0

    # astype
    # Replacement
    op_seriesOp = ops.SeriesOps(
        names="id",
        names_new=None,
        op_names=["astype"],
        dtype="float",
        preprocessing=True,
        replace=True,
    )

    for gdf in data_itr:
        new_gdf = op_seriesOp.apply_op(gdf, columns_ctx, "all")
        assert new_gdf["id"].dtype == "float64"

    # weekday
    # No Replacement
    op_seriesOp = ops.SeriesOps(
        names="timestamp",
        names_new="weekday",
        op_names=["dt", "weekday"],
        preprocessing=True,
        replace=False,
    )

    for gdf in data_itr:
        gdf = gdf[~(gdf["timestamp"].isnull())]
        new_gdf = op_seriesOp.apply_op(gdf, columns_ctx, "all")
        assert np.sum(new_gdf["weekday"] != gdf["timestamp"].dt.weekday) == 0
