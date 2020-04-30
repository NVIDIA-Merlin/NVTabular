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

import nvtabular.ds_iterator as ds
import nvtabular.ops as ops
import nvtabular.preproc as pp
from tests.conftest import allcols_csv, cleanup, mycols_csv, mycols_pq


@cleanup
@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("op_columns", [["x"], None])
def test_gpu_preproc_api(tmpdir, datasets, dump, gpu_memory_frac, engine, op_columns):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = pp.Workflow(
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, to_cpu=False,
    )

    processor.add_feature([ops.ZeroFill(columns=op_columns), ops.LogOp()])
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify())
    processor.finalize()

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    processor.update_stats(data_itr)

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
        assert math.isclose(get_norms(df.y).mean(), processor.stats["means"]["y"], rel_tol=1e-1,)
        assert math.isclose(get_norms(df.y).std(), processor.stats["stds"]["y"], rel_tol=1e-1,)
    assert math.isclose(get_norms(df.x).mean(), processor.stats["means"]["x"], rel_tol=1e-1,)
    assert math.isclose(get_norms(df.x).std(), processor.stats["stds"]["x"], rel_tol=1e-1,)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = processor.stats["encoders"]["name-cat"].get_cats().values_to_string()
        # adding the None entry as a string because of move from gpu
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = processor.stats["encoders"]["name-string"].get_cats().values_to_string()
    # adding the None entry as a string because of move from gpu
    assert cats1 == ["None"] + cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(tmpdir, data_itr, nfiles=10, shuffle=True, apply_ops=True)

    data_itr_2 = ds.GPUDatasetIterator(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"),
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
    )

    df_pp = None
    for chunk in data_itr_2:
        df_pp = cudf.concat([df_pp, chunk], axis=0) if df_pp else chunk

    if engine == "parquet":
        assert df_pp["name-cat"].dtype == "int64"
    assert df_pp["name-string"].dtype == "int64"

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)
    return processor.ds_exports


@pytest.mark.parametrize("batch", [0, 100, 1000])
def test_gpu_file_iterator_parquet(datasets, batch):
    paths = glob.glob(str(datasets["parquet"]) + "/*.parquet")
    df_expect = cudf.read_parquet(paths[0], columns=mycols_pq)
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUFileIterator(
        paths[0], batch_size=batch, gpu_memory_frac=0.01, columns=mycols_pq
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


# THIS Test is commented out because of concat new object issue
# def test_gpu_file_iterator_parquet_row_groups(datasets):
#     paths = glob.glob(str(datasets["parquet"]) + "/*.parquet")
#     df_expect = cudf.read_parquet(paths[0], columns=mycols_pq)
#     df_itr = cudf.DataFrame()
#     data_itr = ds.GPUFileIterator(
#         paths[0], use_row_groups=True, gpu_memory_frac=0.0, columns=mycols_pq
#     )
#     for idx, data_gd in enumerate(data_itr):
#         df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

#     # Make sure the iteration count matches the row-group count
#     (_, num_row_groups, _) = cudf.io.read_parquet_metadata(paths[0])
#     assert num_row_groups == (idx + 1)

#     assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_file_iterator_csv(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUFileIterator(
        paths[0], batch_size=batch, gpu_memory_frac=0.01, columns=mycols_csv, names=names,
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
def test_gpu_dataset_iterator_parquet(datasets, batch):
    paths = glob.glob(str(datasets["parquet"]) + "/*.parquet")
    df_expect = cudf.read_parquet(paths[0], columns=mycols_pq)
    df_expect = cudf.concat([df_expect, cudf.read_parquet(paths[1], columns=mycols_pq)], axis=0)
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUDatasetIterator(
        paths, batch_size=batch, gpu_memory_frac=0.01, columns=mycols_pq
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_dataset_iterator_csv(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    df_expect1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    df_expect2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df_expect = cudf.concat([df_expect1, df_expect2], axis=0)
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()
    data_itr = ds.GPUDatasetIterator(
        paths, batch_size=batch, gpu_memory_frac=0.01, columns=mycols_csv, names=allcols_csv,
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd
    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@cleanup
@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
def test_gpu_preproc(tmpdir, datasets, dump, gpu_memory_frac, engine):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = pp.get_new_config()
    config["FE"]["continuous"] = [ops.ZeroFill()]
    config["PP"]["continuous"] = [[ops.ZeroFill(), ops.Normalize()]]
    config["PP"]["categorical"] = [ops.Categorify()]

    processor = pp.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        to_cpu=False,
    )

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    processor.update_stats(data_itr)
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
        cats0 = processor.stats["encoders"]["name-cat"].get_cats().values_to_string()
        # adding the None entry as a string because of move from gpu
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = processor.stats["encoders"]["name-string"].get_cats().values_to_string()
    # adding the None entry as a string because of move from gpu
    assert cats1 == ["None"] + cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(tmpdir, data_itr, nfiles=10, shuffle=True, apply_ops=True)

    data_itr_2 = ds.GPUDatasetIterator(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"),
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
    )

    df_pp = None
    for chunk in data_itr_2:
        df_pp = cudf.concat([df_pp, chunk], axis=0) if df_pp else chunk

    if engine == "parquet":
        assert df_pp["name-cat"].dtype == "int64"
    assert df_pp["name-string"].dtype == "int64"

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)
    return processor.ds_exports


# @cleanup
# def test_pq_to_pq_processed(tmpdir, datasets):
#     indir = str(datasets["parquet"])
#     outdir = str(tmpdir)
#     cat_names = ["name-cat", "name-string"]
#     columns = mycols_pq
#     cont_names = ["x", "y", "id"]
#     label_name = ["label"]
#     chunk_size = 100

#     config = pp.get_new_config()
#     config["FE"]["continuous"] = [[ops.FillMissing(), ops.LogOp()]]
#     config["PP"]["continuous"] = [[ops.LogOp(), ops.Normalize()]]
#     config["PP"]["categorical"] = [ops.Categorify()]

#     processor = pp.Workflow(
#         cat_names=cat_names,
#         cont_names=cont_names,
#         label_name=label_name,
#         config=config,
#         to_cpu=True,
#     )

#     paths = [os.path.join(indir, x) for x in os.listdir(indir) if x.endswith("parquet")]

#     data_itr = ds.GPUDatasetIterator(paths, use_row_groups=True,)

#     processor.update_stats(data_itr)
#     #     processor.load_stats(sample_stats)
#     processor.pq_to_pq_processed(
#         indir,
#         outdir,
#         columns=mycols_pq,
#         shuffle=True,
#         apply_ops=True,
#         chunk_size=chunk_size,
#     )

#     # TODO: Test that the new parquet dataset is processed correctly
#     old_paths = glob.glob(indir + "/*.parquet")
#     new_paths = glob.glob(outdir + "/*.parquet")
#     assert len(new_paths) == len(old_paths)

#     meta = cudf.io.read_parquet_metadata(outdir + "/_metadata")
#     assert all(x in meta[2] for x in mycols_pq)
#     assert meta[0] // meta[1] <= chunk_size
#     return processor.ds_exports


# This test was removed because the functionality was taken out
# due to an infinite loop behavior that was reported many times.
# def test_estimated_row_size(tmpdir):
#     # Make sure the row_size estimate is what we expect...
#     size = 1000
#     df = cudf.DataFrame(
#         {
#             "int32": np.arange(size, dtype="int32"),
#             "int64": np.arange(size, dtype="int64"),
#             "float64": np.arange(size, dtype="float64"),
#             "str": np.random.choice(["cat", "bat", "dog"], size=size),
#         }
#     )

#     # Write parquet File
#     fn_csv = str(tmpdir) + "/temp.csv"
#     df.to_csv(fn_csv, index=False)

#     # Write csv File
#     df.to_parquet(str(tmpdir))
#     fn_pq = glob.glob(str(tmpdir) + "/*.parquet")[0]

#     # Use PyArrow to get "accurate" in-memory row size
#     read_byte_size = 0
#     for col in cudf.read_parquet(fn_pq)._columns:
#         if col.dtype == "object":
#             max_size = len(max(col)) // 2
#             read_byte_size += int(max_size)
#         else:
#             read_byte_size += col.dtype.itemsize

#     # Check parquet estimate
#     reader_pq = ds.PQFileReader(fn_pq, 0.1, None, row_size=None)
#     estimated_row_size_pq = reader_pq.estimated_row_size
#     assert estimated_row_size_pq == read_byte_size

#     # Check csv estimate
#     reader_csv = ds.CSVFileReader(
#         fn_csv, 0.1, None, row_size=None, dtype=["int32", "int64", "float64", "str"]
#     )
#     estimated_row_size_csv = reader_csv.estimated_row_size
#     assert estimated_row_size_csv == read_byte_size


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("replace", [True, False])
def test_gpu_preproc_config(tmpdir, datasets, dump, gpu_memory_frac, engine, replace):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])

    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=False, names=allcols_csv)[mycols_csv]
    df = cudf.concat([df1, df2], axis=0)
    df["id"] = df["id"].astype("int64")

    if engine == "parquet":
        cat_names = ["name-cat", "name-string"]
        columns = mycols_pq
    else:
        cat_names = ["name-string"]
        columns = mycols_csv
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    config = pp.get_new_config()
    # add operators with dependencies
    config["FE"]["continuous"] = [[ops.FillMissing(replace=replace), ops.LogOp()]]
    config["PP"]["continuous"] = [[ops.LogOp(replace=replace), ops.Normalize()]]
    config["PP"]["categorical"] = [ops.Categorify()]

    processor = pp.Workflow(
        cat_names=cat_names,
        cont_names=cont_names,
        label_name=label_name,
        config=config,
        to_cpu=False,
    )

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    processor.update_stats(data_itr)

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
        get_norms(df.x).mean(), processor.stats["means"]["x" + concat_ops], rel_tol=1e-1,
    )
    assert math.isclose(
        get_norms(df.y).mean(), processor.stats["means"]["y" + concat_ops], rel_tol=1e-1,
    )

    assert math.isclose(
        get_norms(df.x).std(), processor.stats["stds"]["x" + concat_ops], rel_tol=1e-1,
    )
    assert math.isclose(
        get_norms(df.y).std(), processor.stats["stds"]["y" + concat_ops], rel_tol=1e-1,
    )

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = processor.stats["encoders"]["name-cat"].get_cats().values_to_string()
        # adding the None entry as a string because of move from gpu
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = processor.stats["encoders"]["name-string"].get_cats().values_to_string()
    # adding the None entry as a string because of move from gpu
    assert cats1 == ["None"] + cats_expected1

    # Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(tmpdir, data_itr, nfiles=10, shuffle=True, apply_ops=True)

    data_itr_2 = ds.GPUDatasetIterator(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"),
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
    )

    df_pp = None
    for chunk in data_itr_2:
        df_pp = cudf.concat([df_pp, chunk], axis=0) if df_pp else chunk

    if engine == "parquet":
        assert df_pp["name-cat"].dtype == "int64"
    assert df_pp["name-string"].dtype == "int64"

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert num_rows == len(df_pp)
    return processor.ds_exports
