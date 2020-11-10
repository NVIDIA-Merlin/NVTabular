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
import math
import os

import cudf
import numpy as np
import pandas as pd
import pytest
from cudf.tests.utils import assert_eq
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
from nvtabular import ops as ops
from nvtabular.io import Dataset
from tests.conftest import get_cats, mycols_csv


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("op_columns", [["x"], None])
@pytest.mark.parametrize("use_client", [True, False])
def test_gpu_workflow_api(
    tmpdir, client, df, dataset, gpu_memory_frac, engine, dump, op_columns, use_client
):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = nvt.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        client=client if use_client else None,
    )

    processor.add_feature(
        [ops.FillMissing(), ops.Clip(min_value=0, columns=op_columns), ops.LogOp()]
    )
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify(cat_cache="host"))
    processor.finalize()
    assert len(processor.phases) == 2

    processor.update_stats(dataset)

    if dump:
        config_file = tmpdir + "/temp.yaml"
        processor.save_stats(config_file)
        processor.clear_stats()
        processor.load_stats(config_file)

    def get_norms(tar: cudf.Series):
        gdf = tar.fillna(0)
        gdf = gdf * (gdf >= 0).astype("int")
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Clip, Log

    if not op_columns:
        assert math.isclose(get_norms(df.y).mean(), processor.stats["means"]["y"], rel_tol=1e-1)
        assert math.isclose(get_norms(df.y).std(), processor.stats["stds"]["y"], rel_tol=1e-1)
    assert math.isclose(get_norms(df.x).mean(), processor.stats["means"]["x"], rel_tol=1e-1)
    assert math.isclose(get_norms(df.x).std(), processor.stats["stds"]["x"], rel_tol=1e-1)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_host
        cats0 = get_cats(processor, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert cats0.tolist() == [None] + cats_expected0.tolist()
    cats_expected1 = df["name-string"].unique().values_host
    cats1 = get_cats(processor, "name-string")
    # adding the None entry as a string because of move from gpu
    assert cats1.tolist() == [None] + cats_expected1.tolist()

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(
        tmpdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
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

    p = nvt.Workflow(
        cat_names=["ad_id", "source_id", "platform"],
        cont_names=["cont"],
        label_name=["clicked"],
        client=client,
    )
    p.add_feature(ops.FillMissing())
    p.add_feature(ops.Normalize())
    p.add_feature(ops.Categorify())
    p.add_feature(
        ops.TargetEncoding(
            cat_groups=["ad_id", "source_id", "platform"],
            cont_target="clicked",
            kfold=5,
            fold_seed=42,
            p_smooth=20,
        )
    )

    p.apply(nvt.Dataset(gdf_test), record_stats=True)
    assert p.stats


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
def test_gpu_workflow(tmpdir, client, df, dataset, gpu_memory_frac, engine, dump):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["FE"]["continuous"] = [ops.FillMissing(), ops.Clip(min_value=0)]
    config["PP"]["continuous"] = [[ops.FillMissing(), ops.Clip(min_value=0), ops.Normalize()]]
    config["PP"]["categorical"] = [ops.Categorify()]

    processor = nvt.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        client=client,
    )

    processor.update_stats(dataset)
    if dump:
        config_file = tmpdir + "/temp.yaml"
        processor.save_stats(config_file)
        processor.clear_stats()
        processor.load_stats(config_file)

    def get_norms(tar: cudf.Series):
        gdf = tar.fillna(0)
        gdf = gdf * (gdf >= 0).astype("int")
        return gdf

    assert math.isclose(get_norms(df.x).mean(), processor.stats["means"]["x"], rel_tol=1e-4)
    assert math.isclose(get_norms(df.y).mean(), processor.stats["means"]["y"], rel_tol=1e-4)
    assert math.isclose(get_norms(df.x).std(), processor.stats["stds"]["x"], rel_tol=1e-3)
    assert math.isclose(get_norms(df.y).std(), processor.stats["stds"]["y"], rel_tol=1e-3)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_host
        cats0 = get_cats(processor, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert cats0.tolist() == [None] + cats_expected0.tolist()
    cats_expected1 = df["name-string"].unique().values_host
    cats1 = get_cats(processor, "name-string")
    # adding the None entry as a string because of move from gpu
    assert cats1.tolist() == [None] + cats_expected1.tolist()

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(
        tmpdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
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

    processor = nvt.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        client=client,
    )

    processor.add_feature(
        [ops.FillMissing(replace=replace), ops.LogOp(replace=replace), ops.Normalize()]
    )
    processor.add_feature(ops.Categorify())
    processor.finalize()
    assert len(processor.phases) == 2

    processor.update_stats(dataset)

    if dump:
        config_file = tmpdir + "/temp.yaml"
        processor.save_stats(config_file)
        processor.clear_stats()
        processor.load_stats(config_file)

    def get_norms(tar: cudf.Series):
        ser_median = tar.dropna().quantile(0.5, interpolation="linear")
        gdf = tar.fillna(ser_median)
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Clip, Log

    concat_ops = "_FillMissing_1_LogOp_1"
    if replace:
        concat_ops = ""
    assert math.isclose(
        get_norms(df.x).mean(), processor.stats["means"]["x" + concat_ops], rel_tol=1e-1
    )
    assert math.isclose(
        get_norms(df.y).mean(), processor.stats["means"]["y" + concat_ops], rel_tol=1e-1
    )

    assert math.isclose(
        get_norms(df.x).std(), processor.stats["stds"]["x" + concat_ops], rel_tol=1e-1
    )
    assert math.isclose(
        get_norms(df.y).std(), processor.stats["stds"]["y" + concat_ops], rel_tol=1e-1
    )
    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_host
        cats0 = get_cats(processor, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert cats0.tolist() == [None] + cats_expected0.tolist()
    cats_expected1 = df["name-string"].unique().values_host
    cats1 = get_cats(processor, "name-string")
    # adding the None entry as a string because of move from gpu
    assert cats1.tolist() == [None] + cats_expected1.tolist()

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(
        tmpdir, dataset, out_files_per_proc=10, shuffle=nvt.io.Shuffle.PER_PARTITION, apply_ops=True
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
    processor = nvt.Workflow(
        cat_names=[], cont_names=columns, label_name=[], client=client if use_client else None
    )
    processor.add_preprocess(ops.Normalize())
    processor.finalize()
    processor.apply(
        dataset, output_path=out_path, shuffle=shuffle, out_files_per_proc=out_files_per_proc
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
@pytest.mark.parametrize("preproc", ["cat", "cont", "feat", "all"])
def test_join_external_workflow(tmpdir, df, dataset, engine, preproc):

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

    # Define Workflow
    processor = nvt.Workflow(
        cat_names=["name-cat", "name-string"], cont_names=["x", "y", "id"], label_name=["label"]
    )
    if preproc == "cat":
        processor.add_cat_preprocess(merge_op)
    elif preproc == "cont":
        processor.add_cont_preprocess(merge_op)
    elif preproc == "feat":
        processor.add_feature(merge_op)
    else:
        processor.add_preprocess(merge_op)
    processor.finalize()

    processor.apply(dataset, output_format=None)

    # Validate
    for gdf, part in zip(dataset.to_iter(), processor.get_ddf().partitions):
        new_gdf = part.compute(scheduler="synchronous")
        assert len(gdf) == len(new_gdf)
        assert (gdf["id"] + shift).all() == new_gdf["new_col"].all()
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

    workflow = nvt.Workflow(
        cat_names=["cat01"], cont_names=["cont01", "cont02"], label_name=["label"]
    )
    workflow.add_cont_feature(nvt.ops.FillMissing(columns=["cont01"], replace=True))
    workflow.add_cont_preprocess(nvt.ops.NormalizeMinMax(columns=["cont01", "cont02"]))
    workflow.finalize()

    print(df)

    workflow.apply(nvt.Dataset(df), output_path=None)
    result = workflow.get_ddf().compute()
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
    proc = nvt.Workflow(cat_names=["C"], cont_names=["A", "B"], label_name=[])

    proc.add_feature(
        nvt.ops.LambdaOp(op_name="isnull", f=lambda col, gdf: col.isnull(), replace=False)
    )

    proc.add_cat_preprocess(nvt.ops.Categorify())
    train_dataset = nvt.Dataset(gdf, engine="parquet")

    proc.apply(train_dataset, apply_offline=True, record_stats=True, output_path=None)
    result = proc.get_ddf().compute()
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

    proc = nvt.Workflow(
        cat_names=["ad_id", "source_id", "platform"], cont_names=[], label_name=["clicked"]
    )
    # apply dropna
    proc.add_feature(
        [
            nvt.ops.Dropna(["platform"]),
            nvt.ops.JoinGroupby(columns=["ad_id"], cont_names=["clicked"], stats=["sum", "count"]),
            nvt.ops.LambdaOp(
                op_name="ctr",
                f=lambda col, gdf: col / gdf["ad_id_count"],
                columns=["ad_id_clicked_sum"],
                replace=False,
            ),
        ]
    )

    proc.finalize()
    assert len(proc.phases) == 2
    GPU_MEMORY_FRAC = 0.2
    train_dataset = nvt.Dataset(gdf_test, engine="parquet", part_mem_fraction=GPU_MEMORY_FRAC)
    proc.apply(
        train_dataset, apply_offline=True, record_stats=True, output_path=None, shuffle=False
    )
    result = proc.get_ddf().compute()
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

    cont_columns = ["cont1", "cont2"]
    cat_columns = ["cat1", "cat2"]
    label_column = ["label"]

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
    processor = nvt.Workflow(
        cat_names=cat_columns,
        cont_names=cont_columns,
        label_name=label_column,
        client=client if use_client else None,
    )
    processor.add_cont_feature([ops.FillMissing(), ops.Clip(min_value=0), ops.LogOp()])
    processor.add_cat_preprocess(ops.Categorify())

    processor.finalize()
    assert len(processor.phases) == 2
    # Force dtypes
    dict_dtypes = {}
    for col in cont_columns:
        dict_dtypes[col] = np.float32
    for col in cat_columns:
        dict_dtypes[col] = np.float32
    for col in label_column:
        dict_dtypes[col] = np.int64

    if not apply_offline:
        processor.apply(
            dataset,
            output_format=None,
            record_stats=True,
        )
    processor.apply(
        dataset,
        apply_offline=apply_offline,
        record_stats=apply_offline,
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
    cat_names = ["geo_location", "geo_location_country"]
    workflow = nvt.Workflow(cat_names=cat_names, cont_names=[], label_name=[])
    workflow.add_feature(
        [
            ops.LambdaOp(
                op_name="country",
                f=lambda col, gdf: col.str.slice(0, 2),
                columns=["geo_location"],
                replace=False,
            ),
            ops.Categorify(replace=False),
        ]
    )
    workflow.finalize()
    assert len(workflow.phases) == 2

    if use_parquet:
        df.to_parquet(path)
        dataset = nvt.Dataset(path)
    else:
        dataset = nvt.Dataset(df)

    # just make sure this owrks without errors
    workflow.apply(dataset, output_path=out_path)
