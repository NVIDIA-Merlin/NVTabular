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
import shutil

import cudf
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
from nvtabular import ColumnGroup, Dataset, Workflow, ops
from tests.conftest import assert_eq, get_cats, mycols_csv


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("use_client", [True, False])
def test_gpu_workflow_api(tmpdir, client, df, dataset, gpu_memory_frac, engine, dump, use_client):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    norms = ops.Normalize()
    cat_features = cat_names >> ops.Categorify(cat_cache="host")
    cont_features = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp >> norms

    workflow = Workflow(
        cat_features + cont_features + label_name, client=client if use_client else None
    )

    workflow.fit(dataset)

    if dump:
        workflow_dir = os.path.join(tmpdir, "workflow")
        workflow.save(workflow_dir)
        workflow = None

        workflow = Workflow.load(workflow_dir, client=client if use_client else None)

    def get_norms(tar: cudf.Series):
        gdf = tar.fillna(0)
        gdf = gdf * (gdf >= 0).astype("int")
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Clip, Log
    assert math.isclose(get_norms(df.y).mean(), norms.means["y"], rel_tol=1e-1)
    assert math.isclose(get_norms(df.y).std(), norms.stds["y"], rel_tol=1e-1)
    assert math.isclose(get_norms(df.x).mean(), norms.means["x"], rel_tol=1e-1)
    assert math.isclose(get_norms(df.x).std(), norms.stds["x"], rel_tol=1e-1)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_host
        cats0 = get_cats(workflow, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert all(cat in [None] + sorted(cats_expected0.tolist()) for cat in cats0.tolist())
        assert len(cats0.tolist()) == len(cats_expected0.tolist() + [None])
    cats_expected1 = df["name-string"].unique().values_host
    cats1 = get_cats(workflow, "name-string")
    # adding the None entry as a string because of move from gpu
    assert all(cat in [None] + sorted(cats_expected1.tolist()) for cat in cats1.tolist())
    assert len(cats1.tolist()) == len(cats_expected1.tolist() + [None])

    # Write to new "shuffled" and "processed" dataset
    workflow.transform(dataset).to_parquet(
        tmpdir,
        out_files_per_proc=10,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
    )

    dataset_2 = Dataset(glob.glob(str(tmpdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac)

    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)

    if engine == "parquet":
        assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert is_integer_dtype(df_pp["name-string"].dtype)

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)


@pytest.mark.parametrize("engine", ["csv", "csv-no-header"])
def test_gpu_dataset_iterator_csv(df, dataset, engine):
    df_itr = cudf.concat(list(dataset.to_iter(columns=mycols_csv)), axis=0)
    assert_eq(df_itr.reset_index(drop=True), df.reset_index(drop=True))


def test_spec_set(tmpdir, client):
    gdf_test = cudf.DataFrame(
        {
            "ad_id": [1, 2, 2, 6, 6, 8, 3, 3],
            "source_id": [2, 4, 4, 7, 5, 2, 5, 2],
            "platform": [1, 2, np.nan, 2, 1, 3, 3, 1],
            "cont": [1, 2, np.nan, 2, 1, 3, 3, 1],
            "clicked": [1, 0, 1, 0, 0, 1, 1, 0],
        }
    )

    cats = ColumnGroup(["ad_id", "source_id", "platform"])
    cat_features = cats >> ops.Categorify
    cont_features = ColumnGroup(["cont"]) >> ops.FillMissing >> ops.Normalize
    te_features = cats >> ops.TargetEncoding("clicked", kfold=5, fold_seed=42, p_smooth=20)

    p = Workflow(cat_features + cont_features + te_features, client=client)
    p.fit_transform(nvt.Dataset(gdf_test)).to_ddf().compute()


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
def test_gpu_workflow(tmpdir, df, dataset, gpu_memory_frac, engine, dump):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    norms = ops.Normalize()
    conts = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> norms
    cats = cat_names >> ops.Categorify()
    workflow = nvt.Workflow(conts + cats + label_name)

    workflow.fit(dataset)
    if dump:
        workflow_dir = os.path.join(tmpdir, "workflow")
        workflow.save(workflow_dir)
        workflow = None

        workflow = Workflow.load(workflow_dir)

    def get_norms(tar: cudf.Series):
        gdf = tar.fillna(0)
        gdf = gdf * (gdf >= 0).astype("int")
        return gdf

    assert math.isclose(get_norms(df.x).mean(), norms.means["x"], rel_tol=1e-4)
    assert math.isclose(get_norms(df.y).mean(), norms.means["y"], rel_tol=1e-4)
    assert math.isclose(get_norms(df.x).std(), norms.stds["x"], rel_tol=1e-3)
    assert math.isclose(get_norms(df.y).std(), norms.stds["y"], rel_tol=1e-3)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_host
        cats0 = get_cats(workflow, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert all(cat in [None] + sorted(cats_expected0.tolist()) for cat in cats0.tolist())
        assert len(cats0.tolist()) == len(cats_expected0.tolist() + [None])
    cats_expected1 = df["name-string"].unique().values_host
    cats1 = get_cats(workflow, "name-string")
    # adding the None entry as a string because of move from gpu
    assert all(cat in [None] + sorted(cats_expected1.tolist()) for cat in cats1.tolist())
    assert len(cats1.tolist()) == len(cats_expected1.tolist() + [None])

    # Write to new "shuffled" and "processed" dataset
    workflow.transform(dataset).to_parquet(
        output_path=tmpdir, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION
    )

    dataset_2 = Dataset(glob.glob(str(tmpdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac)

    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)

    if engine == "parquet":
        assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert is_integer_dtype(df_pp["name-string"].dtype)

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("replace", [True, False])
def test_gpu_workflow_config(tmpdir, client, df, dataset, gpu_memory_frac, engine, dump, replace):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    norms = ops.Normalize()
    cat_features = cat_names >> ops.Categorify()
    if replace:
        cont_features = cont_names >> ops.FillMissing() >> ops.LogOp >> norms
    else:
        fillmissing_logop = (
            cont_names
            >> ops.FillMissing()
            >> ops.LogOp
            >> ops.Rename(postfix="_FillMissing_1_LogOp_1")
        )
        cont_features = cont_names + fillmissing_logop >> norms

    workflow = Workflow(cat_features + cont_features + label_name, client=client)

    workflow.fit(dataset)

    if dump:
        workflow_dir = os.path.join(tmpdir, "workflow")
        workflow.save(workflow_dir)
        workflow = None

        workflow = Workflow.load(workflow_dir, client=client)

    def get_norms(tar: cudf.Series):
        ser_median = tar.dropna().quantile(0.5, interpolation="linear")
        gdf = tar.fillna(ser_median)
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Clip, Log

    concat_ops = "_FillMissing_1_LogOp_1"
    if replace:
        concat_ops = ""
    assert math.isclose(get_norms(df.x).mean(), norms.means["x" + concat_ops], rel_tol=1e-1)
    assert math.isclose(get_norms(df.y).mean(), norms.means["y" + concat_ops], rel_tol=1e-1)

    assert math.isclose(get_norms(df.x).std(), norms.stds["x" + concat_ops], rel_tol=1e-1)
    assert math.isclose(get_norms(df.y).std(), norms.stds["y" + concat_ops], rel_tol=1e-1)
    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_host
        cats0 = get_cats(workflow, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert all(cat in [None] + sorted(cats_expected0.tolist()) for cat in cats0.tolist())
        assert len(cats0.tolist()) == len(cats_expected0.tolist() + [None])
    cats_expected1 = df["name-string"].unique().values_host
    cats1 = get_cats(workflow, "name-string")
    # adding the None entry as a string because of move from gpu
    assert all(cat in [None] + sorted(cats_expected1.tolist()) for cat in cats1.tolist())
    assert len(cats1.tolist()) == len(cats_expected1.tolist() + [None])

    # Write to new "shuffled" and "processed" dataset
    workflow.transform(dataset).to_parquet(
        tmpdir,
        out_files_per_proc=10,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
    )

    dataset_2 = Dataset(glob.glob(str(tmpdir) + "/*.parquet"), part_mem_fraction=gpu_memory_frac)

    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)

    if engine == "parquet":
        assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert is_integer_dtype(df_pp["name-string"].dtype)

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)


@pytest.mark.parametrize("shuffle", [nvt.io.Shuffle.PER_WORKER, nvt.io.Shuffle.PER_PARTITION, None])
@pytest.mark.parametrize("use_client", [True, False])
def test_parquet_output(client, use_client, tmpdir, shuffle):
    out_files_per_proc = 2
    n_workers = len(client.cluster.workers) if use_client else 1
    out_path = str(tmpdir.mkdir("processed"))
    path = str(tmpdir.join("simple.parquet"))

    size = 25
    row_group_size = 5
    df = pd.DataFrame({"a": np.arange(size)})
    df.to_parquet(path, row_group_size=row_group_size, engine="pyarrow")

    columns = ["a"]
    dataset = nvt.Dataset(path, engine="parquet", row_groups_per_part=1)

    workflow = nvt.Workflow(columns >> ops.Normalize(), client=client if use_client else None)
    workflow.fit_transform(dataset).to_parquet(
        output_path=out_path, shuffle=shuffle, out_files_per_proc=out_files_per_proc
    )

    # Check that the number of output files is correct
    result = glob.glob(os.path.join(out_path, "*.parquet"))
    assert len(result) == out_files_per_proc * n_workers

    # Make sure _metadata exists
    meta_path = os.path.join(out_path, "_metadata")
    assert os.path.exists(meta_path)

    # Make sure _metadata makes sense
    _metadata = cudf.io.read_parquet_metadata(meta_path)
    assert _metadata[0] == size
    assert _metadata[2] == columns


@pytest.mark.parametrize("engine", ["parquet"])
def test_join_external_workflow(tmpdir, df, dataset, engine):

    # Define "external" table
    how = "left"
    drop_duplicates = True
    cache = "device"
    shift = 100
    df_ext = df[["id"]].copy().sort_values("id")
    df_ext["new_col"] = df_ext["id"] + shift
    df_ext["new_col_2"] = "keep"
    df_ext["new_col_3"] = "ignore"
    df_ext_check = df_ext.copy()

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

    # Define Workflow
    gdf = df.reset_index()
    dataset = nvt.Dataset(gdf)
    processor = nvt.Workflow(joined)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute().reset_index()

    # Validate
    check_gdf = gdf.merge(df_ext_check, how=how, on=on)
    assert len(check_gdf) == len(new_gdf)
    assert (new_gdf["id"] + shift).all() == new_gdf["new_col"].all()
    assert gdf["id"].all() == new_gdf["id"].all()
    assert "new_col_2" in new_gdf.columns
    assert "new_col_3" not in new_gdf.columns


def test_chaining_1():
    df = cudf.DataFrame(
        {
            "cont01": np.random.randint(1, 100, 100),
            "cont02": np.random.random(100) * 100,
            "cat01": np.random.randint(0, 10, 100),
            "label": np.random.randint(0, 3, 100),
        }
    )
    df["cont01"][:10] = None

    cont1 = "cont01" >> ops.FillMissing()
    conts = cont1 + "cont02" >> ops.NormalizeMinMax()
    workflow = Workflow(conts + "cat01" + "label")

    result = workflow.fit_transform(Dataset(df)).to_ddf().compute()

    assert result["cont01"].max() <= 1.0
    assert result["cont02"].max() <= 1.0


def test_chaining_2():
    gdf = cudf.DataFrame(
        {
            "A": [1, 2, 2, 9, 6, np.nan, 3],
            "B": [2, np.nan, 4, 7, 7, 2, 5],
            "C": ["a", "b", "c", np.nan, np.nan, "g", "k"],
        }
    )

    cat_names = ["C"]
    cont_names = ["A", "B"]
    label_name = []

    all_features = (
        cat_names + cont_names
        >> ops.LambdaOp(f=lambda col: col.isnull())
        >> ops.Rename(postfix="_isnull")
    )
    cat_features = cat_names >> ops.Categorify()

    workflow = Workflow(all_features + cat_features + label_name)

    dataset = nvt.Dataset(gdf, engine="parquet")

    workflow.fit(dataset)

    result = workflow.transform(dataset).to_ddf().compute()

    assert all(x in list(result.columns) for x in ["A_isnull", "B_isnull", "C_isnull"])
    assert (x in result["C"].unique() for x in set(gdf["C"].dropna().to_arrow()))


def test_chaining_3():
    gdf_test = cudf.DataFrame(
        {
            "ad_id": [1, 2, 2, 6, 6, 8, 3, 3],
            "source_id": [2, 4, 4, 7, 5, 2, 5, 2],
            "platform": [1, 2, np.nan, 2, 1, 3, 3, 1],
            "clicked": [1, 0, 1, 0, 0, 1, 1, 0],
        }
    )

    platform_features = ["platform"] >> ops.Dropna()
    joined = ["ad_id"] >> ops.JoinGroupby(cont_cols=["clicked"], stats=["sum", "count"])
    joined_lambda = (
        joined
        >> ops.LambdaOp(f=lambda col, gdf: col / gdf["ad_id_count"])
        >> ops.Rename(postfix="_ctr")
    )

    workflow = Workflow(platform_features + joined + joined_lambda)

    dataset = nvt.Dataset(gdf_test, engine="parquet")

    workflow.fit(dataset)

    result = workflow.transform(dataset).to_ddf().compute()

    assert all(
        x in result.columns for x in ["ad_id_count", "ad_id_clicked_sum_ctr", "ad_id_clicked_sum"]
    )


@pytest.mark.parametrize("shuffle", [nvt.io.Shuffle.PER_WORKER, nvt.io.Shuffle.PER_PARTITION, None])
@pytest.mark.parametrize("use_client", [True, False])
@pytest.mark.parametrize("apply_offline", [True, False])
def test_workflow_apply(client, use_client, tmpdir, shuffle, apply_offline):
    out_files_per_proc = 2
    out_path = str(tmpdir.mkdir("processed"))
    path = str(tmpdir.join("simple.parquet"))

    size = 25
    row_group_size = 5

    cont_names = ["cont1", "cont2"]
    cat_names = ["cat1", "cat2"]
    label_name = ["label"]

    df = pd.DataFrame(
        {
            "cont1": np.arange(size, dtype=np.float64),
            "cont2": np.arange(size, dtype=np.float64),
            "cat1": np.arange(size, dtype=np.int32),
            "cat2": np.arange(size, dtype=np.int32),
            "label": np.arange(size, dtype=np.float64),
        }
    )
    df.to_parquet(path, row_group_size=row_group_size, engine="pyarrow")

    dataset = nvt.Dataset(path, engine="parquet", row_groups_per_part=1)

    cat_features = cat_names >> ops.Categorify()
    cont_features = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp

    workflow = Workflow(
        cat_features + cont_features + label_name, client=client if use_client else None
    )

    workflow.fit(dataset)

    # Force dtypes
    dict_dtypes = {}
    for col in cont_names:
        dict_dtypes[col] = np.float32
    for col in cat_names:
        dict_dtypes[col] = np.float32
    for col in label_name:
        dict_dtypes[col] = np.int64

    workflow.transform(dataset).to_parquet(
        # apply_offline=apply_offline, Not any more?
        # record_stats=apply_offline, Not any more?
        output_path=out_path,
        shuffle=shuffle,
        out_files_per_proc=out_files_per_proc,
        dtypes=dict_dtypes,
    )

    # Check dtypes
    for filename in glob.glob(os.path.join(out_path, "*.parquet")):
        gdf = cudf.io.read_parquet(filename)
        assert dict(gdf.dtypes) == dict_dtypes


@pytest.mark.parametrize("use_parquet", [True, False])
def test_workflow_generate_columns(tmpdir, use_parquet):
    out_path = str(tmpdir.mkdir("processed"))
    path = str(tmpdir.join("simple.parquet"))

    # Stripped down dataset with geo_locaiton codes like in outbrains
    df = cudf.DataFrame({"geo_location": ["US>CA", "CA>BC", "US>TN>659"]})

    # defining a simple workflow that strips out the country code from the first two digits of the
    # geo_location code and sticks in a new 'geo_location_country' field
    country = (
        ["geo_location"]
        >> ops.LambdaOp(
            f=lambda col: col.str.slice(0, 2),
        )
        >> ops.Rename(postfix="_country")
    )
    cat_features = ["geo_location"] + country >> ops.Categorify()

    workflow = Workflow(cat_features)

    if use_parquet:
        df.to_parquet(path)
        dataset = nvt.Dataset(path)
    else:
        dataset = nvt.Dataset(df)

    # just make sure this works without errors
    workflow.fit(dataset)
    workflow.transform(dataset).to_parquet(out_path)


def test_fit_simple():
    data = cudf.DataFrame({"x": [0, 1, 2, None, 0, 1, 2], "y": [None, 3, 4, 5, 3, 4, 5]})
    dataset = Dataset(data)

    workflow = Workflow(["x", "y"] >> ops.FillMedian() >> (lambda x: x * x))

    workflow.fit(dataset)
    transformed = workflow.transform(dataset).to_ddf().compute()

    expected = cudf.DataFrame({"x": [0, 1, 4, 1, 0, 1, 4], "y": [16, 9, 16, 25, 9, 16, 25]})
    assert_eq(expected, transformed)


def test_transform_geolocation():
    raw = """US>SC>519 US>CA>807 US>MI>505 US>CA>510 CA>NB US>CA>534""".split()
    data = cudf.DataFrame({"geo_location": raw})

    geo_location = ColumnGroup(["geo_location"])
    state = geo_location >> (lambda col: col.str.slice(0, 5)) >> ops.Rename(postfix="_state")
    country = geo_location >> (lambda col: col.str.slice(0, 2)) >> ops.Rename(postfix="_country")
    geo_features = state + country + geo_location >> ops.HashBucket(num_buckets=100)

    # for this workflow we don't have any statoperators, so we can get away without fitting
    workflow = Workflow(geo_features)
    transformed = workflow.transform(Dataset(data)).to_ddf().compute()

    expected = cudf.DataFrame()
    expected["geo_location_state"] = data["geo_location"].str.slice(0, 5).hash_values() % 100
    expected["geo_location_country"] = data["geo_location"].str.slice(0, 2).hash_values() % 100
    expected["geo_location"] = data["geo_location"].hash_values() % 100
    assert_eq(expected, transformed)


def test_workflow_move_saved(tmpdir):
    raw = """US>SC>519 US>CA>807 US>MI>505 US>CA>510 CA>NB US>CA>534""".split()
    data = cudf.DataFrame({"geo": raw})

    geo_location = ColumnGroup(["geo"])
    state = geo_location >> (lambda col: col.str.slice(0, 5)) >> ops.Rename(postfix="_state")
    country = geo_location >> (lambda col: col.str.slice(0, 2)) >> ops.Rename(postfix="_country")
    geo_features = state + country + geo_location >> ops.Categorify()

    # create the workflow and transform the input
    workflow = Workflow(geo_features)
    expected = workflow.fit_transform(Dataset(data)).to_ddf().compute()

    # save the workflow (including categorical mapping parquet files)
    # and then verify we can load the saved workflow after moving the directory
    out_path = os.path.join(tmpdir, "output", "workflow")
    workflow.save(out_path)

    moved_path = os.path.join(tmpdir, "output", "workflow2")
    shutil.move(out_path, moved_path)
    workflow2 = Workflow.load(moved_path)

    # also check that when transforming our input we get the same results after loading
    transformed = workflow2.transform(Dataset(data)).to_ddf().compute()
    assert_eq(expected, transformed)


def test_workflow_input_output_dtypes():
    df = cudf.DataFrame({"genre": ["drama", "comedy"], "user": ["a", "b"], "unneeded": [1, 2]})
    features = [["genre", "user"], "genre"] >> ops.Categorify(encode_type="combo")
    workflow = Workflow(features)
    workflow.fit(Dataset(df))

    assert "unneeded" not in workflow.input_dtypes
    assert set(workflow.input_dtypes.keys()) == {"genre", "user"}
    assert set(workflow.output_dtypes.keys()) == {"genre_user", "genre"}


def test_workflow_transform_ddf_dtypes():
    # Inital Dataset
    df = cudf.datasets.timeseries().reset_index()
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    dataset = Dataset(ddf)

    # Create and Execute Workflow
    cols = ["name", "x", "y", "timestamp"]
    cat_cols = ["id"] >> ops.Normalize()
    workflow = Workflow(cols + cat_cols)
    workflow.fit(dataset)
    transformed_ddf = workflow.transform(dataset).to_ddf()

    # no transforms on the pass through cols, should have original dtypes
    for col in cols:
        assert_eq(ddf.dtypes[col], transformed_ddf.dtypes[col])

    # Followup dask-cudf sorting used to thow an exception  because of dtype issues,
    # check that it works now
    transformed_ddf.sort_values(["id", "timestamp"]).compute()
