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
import dask
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from dask.dataframe import assert_eq
from dask.dataframe.io.demo import names as name_list

import nvtabular as nvt
import nvtabular.io
from nvtabular import ops as ops
from nvtabular.io.parquet import ParquetWriter
from tests.conftest import allcols_csv, mycols_csv, mycols_pq


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
def test_shuffle_gpu(tmpdir, datasets, engine):
    num_files = 2
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    shuf = ParquetWriter(tmpdir, num_out_files=num_files, shuffle=nvt.io.Shuffle.PER_PARTITION)
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
@pytest.mark.parametrize("use_client", [True, False])
def test_hugectr(
    tmpdir, client, df, dataset, output_format, engine, op_columns, num_io_threads, use_client
):
    client = client if use_client else None

    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_names = ["label"]

    # set variables
    nfiles = 10
    ext = ""
    outdir = tmpdir + "/hugectr"
    os.mkdir(outdir)

    # process data
    processor = nvt.Workflow(
        client=client, cat_names=cat_names, cont_names=cont_names, label_name=label_names
    )
    processor.add_feature(
        [
            ops.FillMissing(columns=op_columns),
            ops.Clip(min_value=0, columns=op_columns),
            ops.LogOp(),
        ]
    )
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify())
    processor.finalize()

    # apply the workflow and write out the dataset
    processor.apply(
        dataset,
        output_path=outdir,
        out_files_per_proc=nfiles,
        output_format=output_format,
        shuffle=None,
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
    assert len(data["file_stats"]) == nfiles if not client else nfiles * len(client.cluster.workers)
    for cdata in data["cats"] + data["conts"] + data["labels"]:
        col_summary[cdata["index"]] = cdata["col_name"]

    # Check that data files exist
    ext = ""
    if output_format == "parquet":
        ext = "parquet"
    elif output_format == "hugectr":
        ext = "data"

    data_files = [
        os.path.join(outdir, filename) for filename in os.listdir(outdir) if filename.endswith(ext)
    ]

    # Make sure the columns in "_metadata.json" make sense
    if output_format == "parquet":
        df_check = cudf.read_parquet(os.path.join(outdir, data_files[0]))
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


def test_dataset_partition_shuffle(tmpdir):
    ddf1 = dask.datasets.timeseries(
        start="2000-01-01", end="2000-01-21", freq="1H", dtypes={"name": str, "id": int}
    )
    # Make sure we have enough partitions to ensure
    # random failure is VERY unlikely (prob ~4e-19)
    assert ddf1.npartitions == 20
    columns = list(ddf1.columns)
    ds = nvt.Dataset(ddf1)
    ddf1 = ds.to_ddf()

    # Shuffle
    df1 = ddf1.compute().reset_index(drop=True)
    df2_to_ddf = ds.to_ddf(shuffle=True).compute().reset_index(drop=True)
    df2_to_iter = cudf.concat(list(ds.to_iter(columns=columns, shuffle=True))).reset_index(
        drop=True
    )

    # If we successfully shuffled partitions,
    # our data should not be in the same order
    df3 = df2_to_ddf[["id"]]
    df3["id"] -= df1["id"]
    assert df3["id"].abs().sum() > 0

    # Re-Sort
    df1 = df1.sort_values(columns, ignore_index=True)
    df2_to_ddf = df2_to_ddf.sort_values(columns, ignore_index=True)
    df2_to_iter = df2_to_iter.sort_values(columns, ignore_index=True)

    # Check that the shuffle didn't change the data after re-sorting
    assert_eq(df1, df2_to_ddf)
    assert_eq(df1, df2_to_iter)


@pytest.mark.parametrize("engine", ["csv"])
@pytest.mark.parametrize("num_io_threads", [0, 2])
@pytest.mark.parametrize("nfiles", [0, 1, 2])
@pytest.mark.parametrize("shuffle", [True, False])
def test_mulifile_parquet(tmpdir, dataset, df, engine, num_io_threads, nfiles, shuffle):

    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_names = ["label"]
    columns = cat_names + cont_names + label_names

    outdir = str(tmpdir.mkdir("out"))

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_names)
    processor.finalize()
    processor.apply(
        nvt.Dataset(df),
        output_format="parquet",
        output_path=outdir,
        out_files_per_proc=nfiles,
        num_io_threads=num_io_threads,
        shuffle=shuffle,
    )

    # Check that our output data is exactly the same
    out_paths = glob.glob(os.path.join(outdir, "*.parquet"))
    df_check = cudf.read_parquet(out_paths)
    assert_eq(
        df_check[columns].sort_values(["x", "y"]),
        df[columns].sort_values(["x", "y"]),
        check_index=False,
    )


@pytest.mark.parametrize("freq_threshold", [0, 1, 2])
@pytest.mark.parametrize("shuffle", [nvt.io.Shuffle.PER_PARTITION, None])
@pytest.mark.parametrize("out_files_per_proc", [None, 2])
def test_parquet_lists(tmpdir, freq_threshold, shuffle, out_files_per_proc):
    df = cudf.DataFrame(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )

    input_dir = str(tmpdir.mkdir("input"))
    output_dir = str(tmpdir.mkdir("output"))
    filename = os.path.join(input_dir, "test.parquet")
    df.to_parquet(filename)

    cat_names = ["Authors", "Engaging User"]
    cont_names = []
    label_name = ["Post"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)
    processor.add_preprocess(ops.Categorify(out_path=str(output_dir)))
    processor.finalize()
    processor.apply(
        nvt.Dataset(filename),
        output_format="parquet",
        output_path=output_dir,
        shuffle=shuffle,
        out_files_per_proc=out_files_per_proc,
    )

    out_paths = glob.glob(os.path.join(output_dir, "*.parquet"))
    df_out = cudf.read_parquet(out_paths)
    df_out = df_out.sort_values(by="Post", ascending=True)
    assert df_out["Authors"].to_arrow().to_pylist() == [[1], [1, 4], [2, 3], [3]]


@pytest.mark.parametrize("part_size", [None, "1KB"])
@pytest.mark.parametrize("size", [100, 5000])
@pytest.mark.parametrize("nfiles", [1, 2])
def test_avro_basic(tmpdir, part_size, size, nfiles):

    # Require uavro and fastavro library.
    # Note that fastavro is only required to write
    # avro files for testing, while uavro is actually
    # used by AvroDatasetEngine.
    fa = pytest.importorskip("fastavro")
    pytest.importorskip("uavro")

    # Define avro schema
    schema = fa.parse_schema(
        {
            "name": "avro.example.User",
            "type": "record",
            "fields": [
                {"name": "name", "type": "string"},
                {"name": "age", "type": "int"},
            ],
        }
    )

    # Write avro dataset with two files.
    # Collect block and record (row) count while writing.
    nblocks = 0
    nrecords = 0
    paths = [os.path.join(str(tmpdir), f"test.{i}.avro") for i in range(nfiles)]
    records = []
    for path in paths:
        names = np.random.choice(name_list, size)
        ages = np.random.randint(18, 100, size)
        data = [{"name": names[i], "age": ages[i]} for i in range(size)]
        with open(path, "wb") as f:
            fa.writer(f, schema, data)
        with open(path, "rb") as fo:
            avro_reader = fa.block_reader(fo)
            for block in avro_reader:
                nrecords += block.num_records
                nblocks += 1
                records += list(block)
    if nfiles == 1:
        paths = paths[0]

    # Read back with dask.dataframe
    df = nvt.Dataset(paths, part_size=part_size, engine="avro").to_ddf()

    # Check basic length and partition count
    if part_size == "1KB":
        assert df.npartitions == nblocks
    assert len(df) == nrecords

    # Full comparison
    expect = pd.DataFrame.from_records(records)
    expect["age"] = expect["age"].astype("int32")
    assert_eq(df.compute().reset_index(drop=True), expect)
