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
import os

import cudf
import dask_cudf
import pytest
from dask.dataframe import assert_eq

import nvtabular as nvt
import nvtabular.io
import nvtabular.ops as ops
from nvtabular.io import Shuffler
from tests.conftest import allcols_csv, mycols_csv, mycols_pq


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
def test_shuffle_gpu(tmpdir, datasets, engine):
    num_files = 2
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    shuf = Shuffler(tmpdir, num_files)
    shuf.add_data(df1)
    writer_files = shuf.writer.data_files
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
    if engine == "parquet":
        columns = mycols_pq
    else:
        columns = mycols_csv

    dd = nvtabular.io.Dataset(paths[0], engine=engine, part_mem_fraction=gpu_memory_frac)
    size = 0
    for chunk in dd.to_iter(columns=columns):
        size += chunk.shape[0]

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


@pytest.mark.parametrize("output_format", ["binary", "parquet"])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
def test_hugectr(tmpdir, df, dataset, output_format, engine, op_columns):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_names = ["label"]

    # set variables
    nfiles = 10
    ext = ""
    outdir = tmpdir + "/dontcare"
    h_outdir = tmpdir + "/hugectr"
    os.mkdir(outdir)
    os.mkdir(h_outdir)

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
        shuffle=False,
        hugectr_gen_output=True,
        hugectr_output_path=h_outdir,
        hugectr_num_out_files=nfiles,
        hugectr_output_format=output_format,
    )

    # Check files
    ext = ""
    if output_format == "parquet":
        ext = "parquet"
        assert os.path.isfile(h_outdir + "/metadata.json")
    elif output_format == "binary":
        ext = "data"

    assert os.path.isfile(h_outdir + "/file_list.txt")

    for n in range(nfiles):
        assert os.path.isfile(os.path.join(h_outdir, str(n) + "." + ext))
