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
import json
import math
import os
import warnings
from distutils.version import LooseVersion

import cudf
import dask
import dask.dataframe as dd
import dask_cudf
import numpy as np
import pandas as pd
import pytest
from dask.dataframe import assert_eq
from dask.dataframe.io.demo import names as name_list

import nvtabular as nvt
import nvtabular.io
from nvtabular import ops
from nvtabular.io.parquet import GPUParquetWriter
from tests.conftest import allcols_csv, mycols_csv, mycols_pq, run_in_context


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
def test_shuffle_gpu(tmpdir, datasets, engine):
    num_files = 2
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    shuf = GPUParquetWriter(tmpdir, num_out_files=num_files, shuffle=nvt.io.Shuffle.PER_PARTITION)
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

    size = 0
    ds = nvtabular.io.Dataset(
        paths[0], engine=engine, part_mem_fraction=gpu_memory_frac, dtypes=dtypes
    )
    my_iter = ds.to_iter(columns=columns)
    for chunk in my_iter:
        size += chunk.shape[0]
        assert chunk["id"].dtype == np.int32

    assert size == df1.shape[0]
    assert len(my_iter) == size


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
@pytest.mark.parametrize("num_files", [1, 2])
@pytest.mark.parametrize("cpu", [None, True])
def test_dask_dataset(datasets, engine, num_files, cpu):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    paths = paths[:num_files]
    if engine == "parquet":
        ddf0 = dask_cudf.read_parquet(paths)[mycols_pq]
        dataset = nvtabular.io.Dataset(paths, cpu=cpu)
        result = dataset.to_ddf(columns=mycols_pq)
    else:
        ddf0 = dask_cudf.read_csv(paths, header=None, names=allcols_csv)[mycols_csv]
        dataset = nvtabular.io.Dataset(paths, cpu=cpu, header=None, names=allcols_csv)
        result = dataset.to_ddf(columns=mycols_csv)

    # We do not preserve the index in NVTabular
    if engine == "parquet":
        assert_eq(ddf0, result, check_index=False)
    else:
        assert_eq(ddf0, result)

    # Check that the cpu kwarg is working correctly
    if cpu:
        assert isinstance(result.compute(), pd.DataFrame)

        # Should still work if we move to the GPU
        # (test behavior after repetitive conversion)
        dataset.to_gpu()
        dataset.to_cpu()
        dataset.to_cpu()
        dataset.to_gpu()
        result = dataset.to_ddf()
        assert isinstance(result.compute(), cudf.DataFrame)
    else:
        assert isinstance(result.compute(), cudf.DataFrame)

        # Should still work if we move to the CPU
        # (test behavior after repetitive conversion)
        dataset.to_cpu()
        dataset.to_gpu()
        dataset.to_gpu()
        dataset.to_cpu()
        result = dataset.to_ddf()
        assert isinstance(result.compute(), pd.DataFrame)


@pytest.mark.parametrize("origin", ["cudf", "dask_cudf", "pd", "dd"])
@pytest.mark.parametrize("cpu", [None, True])
def test_dask_dataset_from_dataframe(tmpdir, origin, cpu):

    # Generate a DataFrame-based input
    if origin in ("pd", "dd"):
        df = pd.DataFrame({"a": range(100)})
        if origin == "dd":
            df = dask.dataframe.from_pandas(df, npartitions=4)
    elif origin in ("cudf", "dask_cudf"):
        df = cudf.DataFrame({"a": range(100)})
        if origin == "dask_cudf":
            df = dask_cudf.from_cudf(df, npartitions=4)

    # Convert to an NVTabular Dataset and back to a ddf
    dataset = nvtabular.io.Dataset(df, cpu=cpu)
    result = dataset.to_ddf()

    # Check resulting data
    assert_eq(df, result)

    # Check that the cpu kwarg is working correctly
    if cpu:
        assert isinstance(result.compute(), pd.DataFrame)

        # Should still work if we move to the GPU
        # (test behavior after repetitive conversion)
        dataset.to_gpu()
        dataset.to_cpu()
        dataset.to_cpu()
        dataset.to_gpu()
        result = dataset.to_ddf()
        assert isinstance(result.compute(), cudf.DataFrame)
        dataset.to_cpu()
    else:
        assert isinstance(result.compute(), cudf.DataFrame)

        # Should still work if we move to the CPU
        # (test behavior after repetitive conversion)
        dataset.to_cpu()
        dataset.to_gpu()
        dataset.to_gpu()
        dataset.to_cpu()
        result = dataset.to_ddf()
        assert isinstance(result.compute(), pd.DataFrame)
        dataset.to_gpu()

    # Write to disk and read back
    path = str(tmpdir)
    dataset.to_parquet(path, out_files_per_proc=1, shuffle=None)
    ddf_check = dask_cudf.read_parquet(path).compute()
    if origin in ("dd", "dask_cudf"):
        # Multiple partitions are not guarenteed the same
        # order in output file.
        ddf_check = ddf_check.sort_values("a")
    assert_eq(df, ddf_check, check_index=False)


@pytest.mark.parametrize("cpu", [None, True])
def test_dask_datframe_methods(tmpdir, cpu):
    # Input DataFrame objects
    df1 = cudf.datasets.timeseries(seed=7)[["id", "y"]].iloc[:200]
    df2 = cudf.datasets.timeseries(seed=42)[["id", "x"]].iloc[:100]

    # Initialize and merge Dataset objects
    ds1 = nvtabular.io.Dataset(df1, npartitions=3, cpu=cpu)
    ds2 = nvtabular.io.Dataset(df2, npartitions=2, cpu=not cpu)
    ds3 = nvtabular.io.Dataset.merge(ds1, ds2, on="id", how="inner")

    # Check repartitioning
    ds3 = ds3.repartition(npartitions=4)
    assert ds3.npartitions == 4

    # Check that head, tail, and persist are recognized
    ds1.head()
    ds1.tail()
    ds1.persist()

    # Check merge result
    result = ds3.compute().sort_values(["id", "x", "y"])
    expect = cudf.DataFrame.merge(df1, df2, on="id", how="inner").sort_values(["id", "x", "y"])
    assert_eq(result, expect, check_index=False)


@pytest.mark.parametrize("output_format", ["hugectr", "parquet"])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["x"], None])
@pytest.mark.parametrize("num_io_threads", [0, 2])
@pytest.mark.parametrize("use_client", [True, False])
def test_hugectr(
    tmpdir, client, df, dataset, output_format, engine, op_columns, num_io_threads, use_client
):
    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_names = ["label"]

    # set variables
    nfiles = 10
    ext = ""
    outdir = tmpdir + "/hugectr"
    os.mkdir(outdir)
    outdir = str(outdir)

    conts = nvt.ColumnGroup(cont_names) >> ops.Normalize
    cats = nvt.ColumnGroup(cat_names) >> ops.Categorify
    # We have a global dask client defined in this context, so NVTabular
    # should warn us if we initialze a `Workflow` with `client=None`
    workflow = run_in_context(
        nvt.Workflow,
        conts + cats + label_names,
        context=None if use_client else pytest.warns(UserWarning),
        client=client if use_client else None,
    )
    transformed = workflow.fit_transform(dataset)

    # We have a global dask client defined in this context,
    # so NVTabular should warn us if our `Dataset` was
    # initialized with `client=None`
    if output_format == "hugectr":
        run_in_context(
            transformed.to_hugectr,
            context=None if use_client else pytest.warns(UserWarning),
            cats=cat_names,
            conts=cont_names,
            labels=label_names,
            output_path=outdir,
            out_files_per_proc=nfiles,
            num_threads=num_io_threads,
        )
    else:
        run_in_context(
            transformed.to_parquet,
            context=None if use_client else pytest.warns(UserWarning),
            output_path=outdir,
            out_files_per_proc=nfiles,
            num_threads=num_io_threads,
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
    assert (
        len(data["file_stats"]) == nfiles
        if not use_client
        else nfiles * len(client.cluster.workers)
    )
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
@pytest.mark.parametrize("nfiles", [0, 1, 5])  # Use 5 to test repartition in to_parquet
@pytest.mark.parametrize("shuffle", [nvt.io.Shuffle.PER_WORKER, None])
@pytest.mark.parametrize("file_map", [True, False])
def test_multifile_parquet(tmpdir, dataset, df, engine, num_io_threads, nfiles, shuffle, file_map):

    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y"]
    label_names = ["label"]
    columns = cat_names + cont_names + label_names
    workflow = nvt.Workflow(nvt.ColumnGroup(columns))

    outdir = str(tmpdir.mkdir("out"))
    transformed = workflow.transform(nvt.Dataset(dask_cudf.from_cudf(df, 2)))
    if file_map and nfiles:
        transformed.to_parquet(
            output_path=outdir, num_threads=num_io_threads, shuffle=shuffle, output_files=nfiles
        )
        out_paths = glob.glob(os.path.join(outdir, "part_*"))
        assert len(out_paths) == nfiles
    else:
        transformed.to_parquet(
            output_path=outdir,
            num_threads=num_io_threads,
            shuffle=shuffle,
            out_files_per_proc=nfiles,
        )
        out_paths = glob.glob(os.path.join(outdir, "*.parquet"))

    # Check that our output data is exactly the same
    df_check = cudf.read_parquet(out_paths)
    assert_eq(
        df_check[columns].sort_values(["x", "y"]),
        df[columns].sort_values(["x", "y"]),
        check_index=False,
    )


@pytest.mark.parametrize("output_files", [1, 6, None])
@pytest.mark.parametrize("out_files_per_proc", [None, 4])
@pytest.mark.parametrize("shuffle", [nvt.io.Shuffle.PER_WORKER, False])
def test_to_parquet_output_files(tmpdir, datasets, output_files, out_files_per_proc, shuffle):
    # Simple test to check that the `output_files` and `out_files_per_proc`
    # arguments for `to_parquet` are interacting as expected.
    path = str(datasets["parquet"])
    outdir = str(tmpdir)
    dataset = nvtabular.io.Dataset(path, engine="parquet")
    ddf0 = dataset.to_ddf(columns=mycols_pq)

    if output_files is None:
        # Test expected behavior when a dictionary
        # is specified for output_files
        output_files = {"file.parquet": range(ddf0.npartitions)}

    if isinstance(output_files, dict) and out_files_per_proc:
        # to_parquet should raise an error if we try to
        # use `out_files_per_proc` when a dictionary
        # is passed in for `output_files`
        with pytest.raises(ValueError):
            dataset.to_parquet(
                outdir,
                shuffle=shuffle,
                output_files=output_files,
                out_files_per_proc=out_files_per_proc,
            )
    else:
        # Test normal/correct to_parquet usage
        dataset.to_parquet(
            outdir,
            shuffle=shuffle,
            output_files=output_files,
            out_files_per_proc=out_files_per_proc,
        )

        # Check that the expected number of files has been written
        written_files = glob.glob(os.path.join(outdir, "*.parquet"))
        assert (
            len(written_files) == output_files
            if isinstance(output_files, int)
            else len(output_files)
        )

        # Check that we didn't loose any data
        ddf1 = dd.read_parquet(outdir, columns=mycols_pq)
        assert len(ddf0) == len(ddf1)


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
    cats = cat_names >> ops.Categorify(out_path=str(output_dir))
    workflow = nvt.Workflow(cats + "Post")

    transformed = workflow.fit_transform(nvt.Dataset(filename))
    transformed.to_parquet(
        output_path=output_dir,
        shuffle=shuffle,
        out_files_per_proc=out_files_per_proc,
    )

    out_paths = glob.glob(os.path.join(output_dir, "*.parquet"))
    df_out = cudf.read_parquet(out_paths)
    df_out = df_out.sort_values(by="Post", ascending=True)
    # user C is encoded as 2 because of frequency
    assert df_out["Authors"].to_arrow().to_pylist() == [[1], [1, 4], [3, 2], [2]]


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


@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_validate_dataset(datasets, engine):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
        if engine == "parquet":
            dataset = nvtabular.io.Dataset(str(datasets[engine]), engine=engine)

            # Default file_min_size should result in failed validation
            assert not dataset.validate_dataset()
            assert dataset.validate_dataset(file_min_size=1, require_metadata_file=False)
        else:
            dataset = nvtabular.io.Dataset(paths, header=False, names=allcols_csv)

            # CSV format should always fail validation
            assert not dataset.validate_dataset()


def test_validate_dataset_bad_schema(tmpdir):
    if LooseVersion(dask.__version__) <= "2.30.0":
        # Older versions of Dask will not handle schema mismatch
        pytest.skip("Test requires newer version of Dask.")

    path = str(tmpdir)
    for (fn, df) in [
        ("part.0.parquet", pd.DataFrame({"a": range(10), "b": range(10)})),
        ("part.1.parquet", pd.DataFrame({"a": [None] * 10, "b": range(10)})),
    ]:
        df.to_parquet(os.path.join(path, fn))

    # Initial dataset has mismatched schema and is missing a _metadata file.
    dataset = nvtabular.io.Dataset(path, engine="parquet")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Schema issue should cause validation failure, even if _metadata is ignored
        assert not dataset.validate_dataset(require_metadata_file=False)
        # File size should cause validation error, even if _metadata is generated
        assert not dataset.validate_dataset(add_metadata_file=True)
        # Make sure the last call added a `_metadata` file
        assert len(glob.glob(os.path.join(path, "_metadata")))

        # New datset has a _metadata file, but the file size is still too small
        dataset = nvtabular.io.Dataset(path, engine="parquet")
        assert not dataset.validate_dataset()
        # Ignore file size to get validation success
        assert dataset.validate_dataset(file_min_size=1, row_group_max_size="1GB")


def test_validate_and_regenerate_dataset(tmpdir):

    # Initial timeseries dataset (in cpu memory)
    ddf = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-05",
        freq="60s",
        partition_freq="1d",
        seed=42,
    )
    ds = nvt.Dataset(ddf)

    # Regenerate dataset on disk
    path = str(tmpdir)
    ds.regenerate_dataset(path, part_size="50KiB", file_size="150KiB")

    # Check that the regenerated dataset makes sense.
    # Dataset is ~544KiB - Expect 4 data files
    N = math.ceil(ddf.compute().memory_usage(deep=True).sum() / 150000)
    file_list = glob.glob(os.path.join(path, "*"))
    assert os.path.join(path, "_metadata") in file_list
    assert os.path.join(path, "_file_list.txt") in file_list
    assert os.path.join(path, "_metadata.json") in file_list
    assert len(file_list) == N + 3  # N data files + 3 metadata files

    # Check new dataset validation
    ds2 = nvt.Dataset(path, engine="parquet", part_size="64KiB")
    ds2.validate_dataset(file_min_size=1)

    # Check that dataset content is correct
    assert_eq(ddf, ds2.to_ddf().compute())

    # Check cpu version of `to_ddf`
    assert_eq(ddf, ds2.engine.to_ddf(cpu=True).compute())


@pytest.mark.parametrize("preserve_files", [True, False])
@pytest.mark.parametrize("cpu", [True, False])
def test_dataset_conversion(tmpdir, cpu, preserve_files):

    # Generate toy dataset.
    # Include "hex" strings to mimic Criteo.
    size = 100
    npartitions = 4
    hex_vals = [
        "62770d79",
        "e21f5d58",
        "afea442f",
        "945c7fcf",
        "38b02748",
        "6fcd6dcb",
        "3580aa21",
        "46dedfa6",
    ]
    df = pd.DataFrame(
        {
            "C0": np.random.choice(hex_vals, size),
            "I0": np.random.randint(1_000_000_000, high=10_000_000_000, size=size),
            "F0": np.random.uniform(size=size),
        }
    )
    ddf = dd.from_pandas(df, npartitions=npartitions)

    # Write to csv dataset
    csv_path = os.path.join(str(tmpdir), "csv_dataset")
    ddf.to_csv(csv_path, header=False, sep="\t", index=False)

    # Create NVT Dataset
    dtypes = {"F0": np.float64, "I0": np.int64, "C0": "hex"}
    ds = nvt.Dataset(
        csv_path,
        cpu=cpu,
        engine="csv",
        dtypes=dtypes,
        sep="\t",
        names=["C0", "I0", "F0"],
    )

    # Convert csv dataset to parquet.
    # Adding extra ds -> ds2 step to test `base_dataset` usage.
    pq_path = os.path.join(str(tmpdir), "pq_dataset")
    ds2 = nvt.Dataset(ds.to_ddf(), base_dataset=ds)
    ds2.to_parquet(pq_path, preserve_files=preserve_files, suffix=".pq")

    # Check output.
    # Note that we are converting the inital hex strings to int32.
    ds_check = nvt.Dataset(pq_path, engine="parquet")
    df["C0"] = df["C0"].apply(int, base=16).astype("int32")
    assert_eq(ds_check.to_ddf().compute(), df, check_index=False)

    # Check that the `suffix=".pq"` argument was successful
    assert glob.glob(os.path.join(pq_path, "*.pq"))
    assert not glob.glob(os.path.join(pq_path, "*.parquet"))


@pytest.mark.parametrize("use_file_metadata", [True, None])
@pytest.mark.parametrize("shuffle", [True, False])
def test_parquet_iterator_len(tmpdir, shuffle, use_file_metadata):

    ddf1 = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-6",
        freq="600s",
        partition_freq="1d",
        id_lam=10,
        seed=42,
    ).shuffle("id")

    # Write to parquet dataset
    ddf1.to_parquet(str(tmpdir))

    # Initialize Dataset
    ds = nvt.Dataset(str(tmpdir), engine="parquet")

    # Convert ds -> ds2
    ds2 = nvt.Dataset(ds.to_ddf())

    # Check that iterator lengths match the partition lengths
    ddf2 = ds2.to_ddf(shuffle=shuffle, seed=42)
    for i in range(ddf2.npartitions):
        _iter = ds2.to_iter(
            shuffle=shuffle,
            seed=42,
            indices=[i],
            use_file_metadata=use_file_metadata,
        )
        assert len(ddf2.partitions[i]) == len(_iter)


@pytest.mark.parametrize("cpu", [True, False])
def test_hive_partitioned_data(tmpdir, cpu):

    # Initial timeseries dataset (in cpu memory).
    # Round the full "timestamp" to the hour for partitioning.
    ddf = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-03",
        freq="600s",
        partition_freq="6h",
        seed=42,
    ).reset_index()
    ddf["timestamp"] = ddf["timestamp"].dt.round("D").dt.day
    ds = nvt.Dataset(ddf, engine="parquet")

    # Write the dataset to disk
    path = str(tmpdir)
    partition_keys = ["timestamp", "name"]
    ds.to_parquet(path, partition_on=partition_keys)

    # Make sure the directory structure is hive-like
    df_expect = ddf.compute()
    df_expect = df_expect.sort_values(["id", "x", "y"]).reset_index(drop=True)
    timestamp_check = df_expect["timestamp"].iloc[0]
    name_check = df_expect["name"].iloc[0]
    assert glob.glob(
        os.path.join(
            path,
            f"timestamp={timestamp_check}/name={name_check}/*",
        )
    )

    # Read back with dask.dataframe and check the data
    df_check = dd.read_parquet(path).compute()
    df_check["name"] = df_check["name"].astype("object")
    df_check["timestamp"] = df_check["timestamp"].astype("int64")
    df_check = df_check.sort_values(["id", "x", "y"]).reset_index(drop=True)
    for col in df_expect:
        # Order of columns can change after round-trip partitioning
        assert_eq(df_expect[col], df_check[col], check_index=False)

    # Read back with NVT and check the data
    df_check = nvt.Dataset(path, engine="parquet").to_ddf().compute()
    df_check["name"] = df_check["name"].astype("object")
    df_check["timestamp"] = df_check["timestamp"].astype("int64")
    df_check = df_check.sort_values(["id", "x", "y"]).reset_index(drop=True)
    for col in df_expect:
        # Order of columns can change after round-trip partitioning
        assert_eq(df_expect[col], df_check[col], check_index=False)


@pytest.mark.parametrize("cpu", [True, False])
@pytest.mark.parametrize("partition_on", [None, ["name", "id"], ["name"]])
@pytest.mark.parametrize("keys", [["name"], ["id"], ["name", "id"]])
@pytest.mark.parametrize("npartitions", [None, 2])
def test_dataset_shuffle_on_keys(tmpdir, cpu, partition_on, keys, npartitions):

    # Initial timeseries dataset
    size = 60
    df1 = pd.DataFrame(
        {
            "name": np.random.choice(["Dave", "Zelda"], size=size),
            "id": np.random.choice([0, 1], size=size),
            "x": np.random.uniform(low=0.0, high=10.0, size=size),
            "y": np.random.uniform(low=0.0, high=10.0, size=size),
        }
    )
    ddf1 = dd.from_pandas(df1, npartitions=3)

    # Write the dataset to disk
    path = str(tmpdir)
    ddf1.to_parquet(str(tmpdir), partition_on=partition_on)

    # Construct NVT Dataset
    ds = nvt.Dataset(path, engine="parquet")

    # Shuffle the dataset by `keys`
    ds2 = ds.shuffle_by_keys(keys, npartitions=npartitions)

    # Inspect the result
    ddf2 = ds2.to_ddf()
    if npartitions:
        assert ddf2.npartitions == npartitions

    # A successful shuffle will return the same unique-value
    # count for both the full dask algorithm and a partition-wise sum
    n1 = sum([len(p[keys].drop_duplicates()) for p in ddf2.partitions])
    n2 = len(ddf2[keys].drop_duplicates())
    assert n1 == n2

    # Check that none of the rows was changed
    df1 = df1.sort_values(["id", "x", "y"]).reset_index(drop=True)
    df2 = ddf2.compute().sort_values(["id", "x", "y"]).reset_index(drop=True)
    if partition_on:
        # Dask will convert partitioned columns to Categorical
        df2["name"] = df2["name"].astype("object")
        df2["id"] = df2["id"].astype("int64")
    for col in df1:
        # Order of columns can change after round-trip partitioning
        assert_eq(df1[col], df2[col], check_index=False)
