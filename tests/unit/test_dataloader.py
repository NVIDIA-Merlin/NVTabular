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
import shutil

import cudf
import numpy as np
import pytest
import torch
from cudf.tests.utils import assert_eq
import nvtabular as nvt
import nvtabular.batchloader as bl
import nvtabular.ds_iterator as ds
import nvtabular.ops as ops
import nvtabular.preproc as pp
from tests.conftest import allcols_csv, mycols_csv, mycols_pq


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_file_iterator_ds(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()
    data_itr = bl.FileItrDataset(
        paths[0],
        engine="csv",
        batch_size=batch,
        gpu_memory_frac=0.01,
        columns=mycols_csv,
        names=names,
    )
    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
def test_gpu_file_iterator_dl(datasets, batch, dskey):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    df_itr = cudf.DataFrame()

    processor = pp.Workflow(
        cat_names=["name-string"], cont_names=["x", "y", "id"], label_name=["label"], to_cpu=True,
    )

    data_itr = bl.FileItrDataset(
        paths[0],
        engine="csv",
        batch_size=batch,
        gpu_memory_frac=0.01,
        columns=mycols_csv,
        names=names,
    )

    data_chain = torch.utils.data.ChainDataset([data_itr])
    dlc = bl.DLCollator(processor)
    bl.DLDataLoader(data_itr, collate_fn=dlc.gdf_col, pin_memory=False, num_workers=0)

    for data_gd in data_itr:
        df_itr = cudf.concat([df_itr, data_gd], axis=0) if df_itr else data_gd

    assert len(data_itr) == len(data_chain)
    assert_eq(df_itr.reset_index(drop=True), df_expect.reset_index(drop=True))


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
def test_shuffle_gpu(tmpdir, datasets, engine):
    num_files = 2
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    shuf = ds.Shuffler(tmpdir, num_files)
    shuf.add_data(df1)
    writer_files = shuf.writer_files
    shuf.close()
    if engine == "parquet":
        df3 = cudf.read_parquet(writer_files[0])[mycols_pq]
        df4 = cudf.read_parquet(writer_files[1])[mycols_pq]
    else:
        df3 = cudf.read_parquet(writer_files[0])[mycols_csv]
        df4 = cudf.read_parquet(writer_files[1])[mycols_csv]
    assert df1.shape[0] == df3.shape[0] + df4.shape[0]


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("dump", [True, False])
@pytest.mark.parametrize("preprocessing", [True, False])
def test_gpu_preproc(tmpdir, datasets, dump, gpu_memory_frac, engine, preprocessing):
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
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, to_cpu=True,
    )

    processor.add_feature([ops.FillMissing(), ops.LogOp(preprocessing=preprocessing)])
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
        ser_median = tar.dropna().quantile(0.5, interpolation="linear")
        gdf = tar.fillna(ser_median)
        gdf = np.log(gdf + 1)
        return gdf

    # Check mean and std - No good right now we have to add all other changes; Zerofill, Log
    x_col = "x" if preprocessing else "x_LogOp"
    y_col = "y" if preprocessing else "y_LogOp"
    assert math.isclose(get_norms(df.x).mean(), processor.stats["means"][x_col], rel_tol=1e-2,)
    assert math.isclose(get_norms(df.y).mean(), processor.stats["means"][y_col], rel_tol=1e-2,)
    assert math.isclose(get_norms(df.x).std(), processor.stats["stds"][x_col], rel_tol=1e-2,)
    assert math.isclose(get_norms(df.y).std(), processor.stats["stds"][y_col], rel_tol=1e-2,)

    # Check median (TODO: Improve the accuracy)
    x_median = df.x.dropna().quantile(0.5, interpolation="linear")
    y_median = df.y.dropna().quantile(0.5, interpolation="linear")
    id_median = df.id.dropna().quantile(0.5, interpolation="linear")
    assert math.isclose(x_median, processor.stats["medians"]["x"], rel_tol=1e1)
    assert math.isclose(y_median, processor.stats["medians"]["y"], rel_tol=1e1)
    assert math.isclose(id_median, processor.stats["medians"]["id"], rel_tol=1e1)

    # Check that categories match
    if engine == "parquet":
        cats_expected0 = df["name-cat"].unique().values_to_string()
        cats0 = processor.stats["encoders"]["name-cat"].get_cats().values_to_string()
        assert cats0 == ["None"] + cats_expected0
    cats_expected1 = df["name-string"].unique().values_to_string()
    cats1 = processor.stats["encoders"]["name-string"].get_cats().values_to_string()
    print(cats1)
    assert cats1 == ["None"] + cats_expected1

    #     Write to new "shuffled" and "processed" dataset
    processor.write_to_dataset(tmpdir, data_itr, nfiles=10, shuffle=True, apply_ops=True)

    processor.create_final_cols()

    # if preprocessing
    if not preprocessing:
        for col in cont_names:
            assert f"{col}_LogOp" in processor.columns_ctx["final"]["cols"]["continuous"]

    dlc = bl.DLCollator(preproc=processor, apply_ops=False)
    data_files = [
        bl.FileItrDataset(
            x, use_row_groups=True, gpu_memory_frac=gpu_memory_frac, names=allcols_csv,
        )
        for x in glob.glob(str(tmpdir) + "/ds_part.*.parquet")
    ]

    data_itr = torch.utils.data.ChainDataset(data_files)
    dl = bl.DLDataLoader(data_itr, collate_fn=dlc.gdf_col, pin_memory=False, num_workers=0)

    len_df_pp = 0
    for chunk in dl:
        len_df_pp += len(chunk[0][0])

    data_itr = ds.GPUDatasetIterator(
        glob.glob(str(tmpdir) + "/ds_part.*.parquet"),
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    x = processor.ds_to_tensors(data_itr, apply_ops=False)

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(str(tmpdir) + "/_metadata")
    assert len(x[0]) == len_df_pp

    itr_ds = bl.TensorItrDataset([x[0], x[1], x[2]], batch_size=512000)
    count_tens_itr = 0
    for data_gd in itr_ds:
        count_tens_itr += len(data_gd[1])
        assert data_gd[0][0].shape[1] > 0
        assert data_gd[0][1].shape[1] > 0

    assert len_df_pp == count_tens_itr
    if os.path.exists(processor.ds_exports):
        shutil.rmtree(processor.ds_exports)


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("batch_size", [1, 10, 100])
def test_gpu_dl(tmpdir, datasets, batch_size, gpu_memory_frac, engine):
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
        cat_names=cat_names, cont_names=cont_names, label_name=label_name, to_cpu=True,
    )

    processor.add_feature([ops.FillMissing()])
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify())

    data_itr = ds.GPUDatasetIterator(
        paths,
        columns=columns,
        use_row_groups=True,
        gpu_memory_frac=gpu_memory_frac,
        names=allcols_csv,
    )

    output_train = os.path.join(tmpdir, "train/")
    os.mkdir(output_train)

    processor.apply(
        data_itr,
        apply_offline=True,
        record_stats=True,
        shuffle=True,
        output_path=output_train,
        num_out_files=2,
    )

    tar_paths = [
        os.path.join(output_train, x) for x in os.listdir(output_train) if x.endswith("parquet")
    ]

    data_itr = nvt.batchloader.TorchTensorBatchDatasetItr(
        tar_paths[0],
        engine="parquet",
        sub_batch_size=batch_size,
        gpu_memory_frac=gpu_memory_frac,
        cats=cat_names,
        conts=cont_names,
        labels=["label"],
        names=mycols_csv,
        sep="\t",
    )

    num_rows, num_row_groups, col_names = cudf.io.read_parquet_metadata(tar_paths[0])
    rows = 0
    for idx, chunk in enumerate(data_itr):
        rows += len(chunk[0])
        del chunk

    # accounts for incomplete batches at the end of chunks
    # that dont necesssarily have the full batch_size
    assert (idx + 1) * batch_size >= rows
    assert rows == num_rows
    if os.path.exists(output_train):
        shutil.rmtree(output_train)
