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
import json
import os

import cudf
import dask_cudf
import numpy as np
import pytest
from dask.dataframe import assert_eq

import nvtabular as nvt
import nvtabular.io
import nvtabular.ops as ops
from nvtabular.io import ParquetWriter
from tests.conftest import allcols_csv, mycols_csv, mycols_pq


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
def test_shuffle_gpu(tmpdir, datasets, engine):
    num_files = 2
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    shuf = ParquetWriter(tmpdir, num_out_files=num_files, shuffle="partial")
    shuf.add_data(df1)
    writer_files = shuf.data_paths
    shuf.close()
    if engine == "parquet":
        df3 = cudf.read_parquet(writer_files[0])[mycols_pq]
        df4 = cudf.read_parquet(writer_files[1])[mycols_pq]
    else:
        df3 = cudf.read_parquet(writer_files[0])[mycols_csv]
        df4 = cudf.read_parquet(writer_files[1])[mycols_csv]
    assert df1.shape[0] == df3.shape[0] + df4.shape[0]


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_dask_dataset_itr(tmpdir, datasets, engine, gpu_memory_frac):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=0, names=allcols_csv)[mycols_csv]
    dtypes = {"id": np.int32}
    if engine == "parquet":
        columns = mycols_pq
    else:
        columns = mycols_csv

    dd = nvtabular.io.Dataset(
        paths[0], engine=engine, part_mem_fraction=gpu_memory_frac, dtypes=dtypes
    )
    size = 0
    for chunk in dd.to_iter(columns=columns):
        size += chunk.shape[0]
        assert chunk["id"].dtype == np.int32

    assert size == df1.shape[0]


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
@pytest.mark.parametrize("num_files", [1, 2])
def test_dask_dataset(datasets, engine, num_files):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    paths = paths[:num_files]
    if engine == "parquet":
        ddf0 = dask_cudf.read_parquet(paths)[mycols_pq]
        dataset = nvtabular.io.Dataset(paths)
        result = dataset.to_ddf(columns=mycols_pq)
    else:
        ddf0 = dask_cudf.read_csv(paths, header=False, names=allcols_csv)[mycols_csv]
        dataset = nvtabular.io.Dataset(paths, header=False, names=allcols_csv)
        result = dataset.to_ddf(columns=mycols_csv)

    assert_eq(ddf0, result)


@pytest.mark.parametrize("output_format", ["hugectr", "parquet"])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
@pytest.mark.parametrize("num_io_threads", [0, 2])
def test_hugectr(tmpdir, df, dataset, output_format, engine, op_columns, num_io_threads):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_names = ["label"]

    # set variables
    nfiles = 10
    ext = ""
    outdir = tmpdir + "/hugectr"
    os.mkdir(outdir)

    # process data
    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_names)
    processor.add_feature([ops.ZeroFill(columns=op_columns), ops.LogOp()])
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify())
    processor.finalize()

    # Need to collect statistics first (for now)
    processor.update_stats(dataset)

    # Second "online" pass to write HugeCTR output
    processor.apply(
        dataset,
        apply_offline=False,
        record_stats=False,
        output_path=outdir,
        out_files_per_proc=nfiles,
        output_format=output_format,
        shuffle=False,
        num_io_threads=num_io_threads,
    )

    # Check for _file_list.txt
    assert os.path.isfile(outdir + "/_file_list.txt")

    # Check for _metadata.json
    assert os.path.isfile(outdir + "/_metadata.json")

    # Check contents of _metadata.json
    data = {}
    col_summary = {}
    with open(outdir + "/_metadata.json", "r") as fil:
        for k, v in json.load(fil).items():
            data[k] = v
    assert "cats" in data
    assert "conts" in data
    assert "labels" in data
    assert "file_stats" in data
    assert len(data["file_stats"]) == nfiles
    for cdata in data["cats"] + data["conts"] + data["labels"]:
        col_summary[cdata["index"]] = cdata["col_name"]

    # Check that data files exist
    ext = ""
    if output_format == "parquet":
        ext = "parquet"
    elif output_format == "hugectr":
        ext = "data"
    for n in range(nfiles):
        assert os.path.isfile(os.path.join(outdir, str(n) + "." + ext))

    # Make sure the columns in "_metadata.json" make sense
    if output_format == "parquet":
        df_check = cudf.read_parquet(os.path.join(outdir, "0.parquet"))
        for i, name in enumerate(df_check.columns):
            if i in col_summary:
                assert col_summary[i] == name


@pytest.mark.parametrize("inp_format", ["dask", "dask_cudf", "cudf", "pandas"])
def test_ddf_dataset_itr(tmpdir, datasets, inp_format):
    paths = glob.glob(str(datasets["parquet"]) + "/*." + "parquet".split("-")[0])
    ddf1 = dask_cudf.read_parquet(paths)[mycols_pq]
    df1 = ddf1.compute()
    if inp_format == "dask":
        ds = nvtabular.io.Dataset(ddf1.to_dask_dataframe())
    elif inp_format == "dask_cudf":
        ds = nvtabular.io.Dataset(ddf1)
    elif inp_format == "cudf":
        ds = nvtabular.io.Dataset(df1)
    elif inp_format == "pandas":
        ds = nvtabular.io.Dataset(df1.to_pandas())
    assert_eq(df1, cudf.concat(list(ds.to_iter(columns=mycols_pq))))
