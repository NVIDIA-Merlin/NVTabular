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

import cudf
import numpy as np
import pytest
from cudf.tests.utils import assert_eq
from pandas.api.types import is_integer_dtype

import nvtabular as nvt
import nvtabular.ops as ops
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

    processor.add_feature([ops.ZeroFill(columns=op_columns), ops.LogOp()])
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify(cat_cache="host"))
    processor.finalize()

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

    # Check mean and std - No good right now we have to add all other changes; Zerofill, Log

    if not op_columns:
        assert math.isclose(get_norms(df.y).mean(), processor.stats["means"]["y"], rel_tol=1e-1)
        assert math.isclose(get_norms(df.y).std(), processor.stats["stds"]["y"], rel_tol=1e-1)
    assert math.isclose(get_norms(df.x).mean(), processor.stats["means"]["x"], rel_tol=1e-1)
    assert math.isclose(get_norms(df.x).std(), processor.stats["stds"]["x"], rel_tol=1e-1)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = get_cats(processor, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = get_cats(processor, "name-string")
    # adding the None entry as a string because of move from gpu
    assert cats1 == ["None"] + cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(tmpdir, dataset, nfiles=10, shuffle=True, apply_ops=True)

    dataset_2 = Dataset(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"), part_mem_fraction=gpu_memory_frac,
    )

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


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
def test_gpu_workflow(tmpdir, client, df, dataset, gpu_memory_frac, engine, dump):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = nvt.workflow.get_new_config()
    config["FE"]["continuous"] = [ops.ZeroFill()]
    config["PP"]["continuous"] = [[ops.ZeroFill(), ops.Normalize()]]
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
    #     assert math.isclose(get_norms(df.id).mean(),
    #                         processor.stats["means"]["id_ZeroFill_LogOp"], rel_tol=1e-4)
    assert math.isclose(get_norms(df.x).std(), processor.stats["stds"]["x"], rel_tol=1e-3)
    assert math.isclose(get_norms(df.y).std(), processor.stats["stds"]["y"], rel_tol=1e-3)
    #     assert math.isclose(get_norms(df.id).std(),
    #                         processor.stats["stds"]["id_ZeroFill_LogOp"], rel_tol=1e-3)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = get_cats(processor, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = get_cats(processor, "name-string")
    # adding the None entry as a string because of move from gpu
    assert cats1 == ["None"] + cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(tmpdir, dataset, nfiles=10, shuffle=True, apply_ops=True)

    dataset_2 = Dataset(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"), part_mem_fraction=gpu_memory_frac,
    )

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

    config = nvt.workflow.get_new_config()
    # add operators with dependencies
    config["FE"]["continuous"] = [[ops.FillMissing(replace=replace), ops.LogOp(replace=replace)]]
    config["PP"]["continuous"] = [[ops.LogOp(replace=replace), ops.Normalize()]]
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
        ser_median = tar.dropna().quantile(0.5, interpolation="linear")
        gdf = tar.fillna(ser_median)
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Zerofill, Log

    concat_ops = "_FillMissing_LogOp"
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
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = get_cats(processor, "name-cat")
        # adding the None entry as a string because of move from gpu
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = get_cats(processor, "name-string")
    # adding the None entry as a string because of move from gpu
    assert cats1 == ["None"] + cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(tmpdir, dataset, nfiles=10, shuffle=True, apply_ops=True)

    dataset_2 = Dataset(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"), part_mem_fraction=gpu_memory_frac,
    )

    df_pp = cudf.concat(list(dataset_2.to_iter()), axis=0)

    if engine == "parquet":
        assert is_integer_dtype(df_pp["name-cat"].dtype)
    assert is_integer_dtype(df_pp["name-string"].dtype)

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)
