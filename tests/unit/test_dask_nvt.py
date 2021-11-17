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

import glob
import math
import os

import pandas as pd
import pytest
from dask.dataframe import assert_eq
from dask.dataframe import from_pandas as dd_from_pandas
from dask.dataframe import read_parquet as dd_read_parquet

from nvtabular import ColumnSelector, Dataset, Workflow, ops
from nvtabular.io.shuffle import Shuffle
from tests.conftest import allcols_csv, mycols_csv, mycols_pq, run_in_context

cudf = pytest.importorskip("cudf")
dask_cudf = pytest.importorskip("dask_cudf")

# Dummy operator logic to test stats
# TODO: Possibly add public API to add
#       standalone Stat Ops


def _dummy_op_logic(gdf, target_columns, _id="dummy", **kwargs):
    cont_names = target_columns
    if not cont_names:
        return gdf
    new_gdf = gdf[cont_names]
    new_cols = [f"{col}_{_id}" for col in new_gdf.columns]
    new_gdf.columns = new_cols
    return new_gdf


@pytest.mark.parametrize("part_mem_fraction", [0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("freq_threshold", [0, 150])
@pytest.mark.parametrize("cat_cache", ["device", None])
@pytest.mark.parametrize("on_host", [True, False])
@pytest.mark.parametrize("shuffle", [Shuffle.PER_WORKER, None])
@pytest.mark.parametrize("cpu", [True, False])
def test_dask_workflow_api_dlrm(
    client,
    tmpdir,
    datasets,
    freq_threshold,
    part_mem_fraction,
    engine,
    cat_cache,
    on_host,
    shuffle,
    cpu,
):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    paths = sorted(paths)
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    elif engine == "csv":
        df1 = cudf.read_csv(paths[0], header=0)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=0)[mycols_csv]
    else:
        df1 = cudf.read_csv(paths[0], names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], names=allcols_csv)[mycols_csv]
    df0 = cudf.concat([df1, df2], axis=0)
    df0 = df0.to_pandas() if cpu else df0

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
    else:
        cat_names = ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    cats = cat_names >> ops.Categorify(
        freq_threshold=freq_threshold, out_path=str(tmpdir), cat_cache=cat_cache, on_host=on_host
    )

    conts = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()

    workflow = Workflow(cats + conts + label_name, client=client)

    if engine in ("parquet", "csv"):
        dataset = Dataset(paths, cpu=cpu, part_mem_fraction=part_mem_fraction)
    else:
        dataset = Dataset(paths, cpu=cpu, names=allcols_csv, part_mem_fraction=part_mem_fraction)

    output_path = os.path.join(tmpdir, "processed")

    transformed = workflow.fit_transform(dataset)
    transformed.to_parquet(output_path=output_path, shuffle=shuffle, out_files_per_proc=1)

    result = transformed.to_ddf().compute()
    assert len(df0) == len(result)
    assert result["x"].min() == 0.0
    assert result["x"].isna().sum() == 0
    assert result["y"].min() == 0.0
    assert result["y"].isna().sum() == 0

    # Check categories.  Need to sort first to make sure we are comparing
    # "apples to apples"
    expect = df0.sort_values(["label", "x", "y", "id"]).reset_index(drop=True).reset_index()
    got = result.sort_values(["label", "x", "y", "id"]).reset_index(drop=True).reset_index()
    dfm = expect.merge(got, on="index", how="inner")[["name-string_x", "name-string_y"]]
    dfm_gb = dfm.groupby(["name-string_x", "name-string_y"]).agg(
        {"name-string_x": "count", "name-string_y": "count"}
    )
    if freq_threshold:
        dfm_gb = dfm_gb[dfm_gb["name-string_x"] >= freq_threshold]
    assert_eq(dfm_gb["name-string_x"], dfm_gb["name-string_y"], check_names=False)

    # Read back from disk
    if cpu:
        df_disk = dd_read_parquet(output_path).compute()
    else:
        df_disk = dask_cudf.read_parquet(output_path).compute()

    # we don't have a deterministic ordering here, especially when using
    # a dask client with multiple workers - so we need to sort the values here
    columns = ["label", "x", "y", "id"] + cat_names
    got = result.sort_values(columns).reset_index(drop=True)
    expect = df_disk.sort_values(columns).reset_index(drop=True)
    assert_eq(got, expect)


@pytest.mark.parametrize("part_mem_fraction", [0.01])
def test_dask_groupby_stats(client, tmpdir, datasets, part_mem_fraction):

    engine = "parquet"
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    df1 = cudf.read_parquet(paths[0])[mycols_pq]
    df2 = cudf.read_parquet(paths[1])[mycols_pq]
    df0 = cudf.concat([df1, df2], axis=0)

    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    features = cat_names >> ops.JoinGroupby(
        cont_cols=cont_names, stats=["count", "sum", "std", "min"], out_path=str(tmpdir)
    )

    dataset = Dataset(paths, part_mem_fraction=part_mem_fraction)
    workflow = Workflow(features + cat_names + cont_names + label_name, client=client)
    result = workflow.fit_transform(dataset).to_ddf().compute(scheduler="synchronous")

    # Validate result
    assert len(df0) == len(result)
    assert "name-cat_x_std" in result.columns
    assert "name-cat_x_var" not in result.columns
    assert "name-string_x_std" in result.columns
    assert "name-string_x_var" not in result.columns

    # Check results.  Need to sort for direct comparison
    expect = df0.sort_values(["label", "x", "y", "id"]).reset_index(drop=True).reset_index()
    got = result.sort_values(["label", "x", "y", "id"]).reset_index(drop=True).reset_index()
    gb_e = expect.groupby("name-cat").aggregate({"name-cat": "count", "x": ["sum", "min", "std"]})
    gb_e.columns = ["count", "sum", "min", "std"]
    df_check = got.merge(gb_e, left_on="name-cat", right_index=True, how="left")
    assert_eq(df_check["name-cat_count"], df_check["count"].astype("int64"), check_names=False)
    assert_eq(df_check["name-cat_x_sum"], df_check["sum"], check_names=False)
    assert_eq(df_check["name-cat_x_min"], df_check["min"], check_names=False)
    assert_eq(df_check["name-cat_x_std"], df_check["std"], check_names=False)


@pytest.mark.parametrize("part_mem_fraction", [0.01])
@pytest.mark.parametrize("use_client", [True, False])
def test_cats_and_groupby_stats(client, tmpdir, datasets, part_mem_fraction, use_client):
    engine = "parquet"
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]

    cats = ColumnSelector(cat_names)
    cat_features = cats >> ops.Categorify(out_path=str(tmpdir), freq_threshold=10, on_host=True)
    groupby_features = cats >> ops.JoinGroupby(
        cont_cols=cont_names, stats=["count", "sum"], out_path=str(tmpdir)
    )

    # We have a global dask client defined in this context, so NVTabular
    # should warn us if we initialize a `Workflow` with `client=None`
    workflow = run_in_context(
        Workflow,
        cat_features + groupby_features,
        context=None if use_client else pytest.warns(UserWarning),
        client=client if use_client else None,
    )
    dataset = Dataset(paths, part_mem_fraction=part_mem_fraction)
    result = workflow.fit_transform(dataset).to_ddf().compute()

    assert "name-cat_x_sum" in result.columns
    assert "name-string_x_sum" in result.columns


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("cpu", [None, True])
def test_dask_normalize(client, tmpdir, datasets, engine, cpu):

    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    df1 = cudf.read_parquet(paths[0])[mycols_pq]
    df2 = cudf.read_parquet(paths[1])[mycols_pq]
    df0 = cudf.concat([df1, df2], axis=0)

    cat_names = ["name-cat", "name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    normalize = ops.Normalize()
    conts = cont_names >> ops.FillMissing() >> normalize
    workflow = Workflow(conts + cat_names + label_name, client=client)

    dataset = Dataset(paths, engine=engine, cpu=cpu)
    result = workflow.fit_transform(dataset).to_ddf().compute()

    # Make sure we collected accurate statistics
    means = df0[cont_names].mean()
    stds = df0[cont_names].std()
    for name in cont_names:
        assert math.isclose(means[name], normalize.means[name], rel_tol=1e-3)
        assert math.isclose(stds[name], normalize.stds[name], rel_tol=1e-3)

    # New (normalized) means should all be close to zero
    new_means = result[cont_names].mean()
    for name in cont_names:
        assert new_means[name] < 1e-3


@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("shuffle", [Shuffle.PER_WORKER, None])
@pytest.mark.parametrize("cpu", [None, True])
def test_dask_preproc_cpu(client, tmpdir, datasets, engine, shuffle, cpu):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    elif engine == "csv":
        df1 = cudf.read_csv(paths[0], header=0)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=0)[mycols_csv]
    else:
        df1 = cudf.read_csv(paths[0], names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], names=allcols_csv)[mycols_csv]
    df0 = cudf.concat([df1, df2], axis=0)

    if engine in ("parquet", "csv"):
        dataset = Dataset(paths, part_size="1MB", cpu=cpu)
    else:
        dataset = Dataset(paths, names=allcols_csv, part_size="1MB", cpu=cpu)

    # Simple transform (normalize)
    cat_names = ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]
    conts = cont_names >> ops.FillMissing() >> ops.Normalize()
    workflow = Workflow(conts + cat_names + label_name, client=client)
    transformed = workflow.fit_transform(dataset)

    # Write out dataset
    output_path = os.path.join(tmpdir, "processed")
    transformed.to_parquet(output_path=output_path, shuffle=shuffle, out_files_per_proc=4)

    # Check the final result
    df_disk = dd_read_parquet(output_path, engine="pyarrow").compute()
    assert_eq(
        df0.sort_values(["id", "x"])[["name-string", "label"]],
        df_disk.sort_values(["id", "x"])[["name-string", "label"]],
        check_index=False,
    )


@pytest.mark.parametrize("cpu", [None, True])
def test_filtered_partition(tmpdir, cpu):
    # Toy DataFrame example
    df = pd.DataFrame({"col": range(100)})
    ddf = dd_from_pandas(df, npartitions=5)
    dataset = Dataset(ddf, cpu=cpu)

    # Workflow
    filtered = ["col"] >> ops.Filter(lambda df: df["col"] < 75)
    workflow = Workflow(filtered)

    # Write result to disk
    workflow.transform(dataset).to_parquet(str(tmpdir))
