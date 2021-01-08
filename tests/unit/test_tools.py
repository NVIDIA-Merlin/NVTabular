import glob
import json
import os

import fsspec
import pytest
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular as nvt
from nvtabular.io import Dataset


@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_inspect(tmpdir, datasets, engine):
    # Dataset
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    output_file = tmpdir + "/dataset_info.json"

    # Dataset columns type config
    columns_dict = {}
    columns_dict["cats"] = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    columns_dict["cats_mh"] = []
    columns_dict["conts"] = ["x", "y"]
    columns_dict["labels"] = ["label"]
    all_cols = (
        columns_dict["cats"]
        + columns_dict["cats_mh"]
        + columns_dict["conts"]
        + columns_dict["labels"]
    )

    # Create inspector and inspect
    a = nvt.tools.DatasetInspector()
    a.inspect(paths, engine, columns_dict, output_file)

    # Check output_file was created
    assert os.path.isfile(output_file)

    # Read output file
    with fsspec.open(output_file) as f:
        output = json.load(f)

    # Get ddf and cluster to check
    dataset = Dataset(paths, engine=engine)
    ddf = dataset.to_ddf()
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # Check conts
    for col in all_cols:
        # Check dtype for all
        # TODO: Fix Dask-cudf _meta
        if col in columns_dict["cats"]:
            assert output[col]["dtype"] == "string"
            ddf[col] = ddf[col].map_partitions(lambda x: x.str.len())
            ddf.compute()
        else:
            assert output[col]["dtype"] == str(ddf[col].dtype)
        # Check percentage of nan for all
        assert output[col]["nans_%"] == (100 * (1 - ddf[col].count().compute() / len(ddf[col])))
        # Check max/min/mean for all
        if col not in columns_dict["labels"]:
            assert output[col]["min"] == ddf[col].min().compute()
            assert output[col]["max"] == ddf[col].max().compute()
            if col in columns_dict["conts"]:
                assert output[col]["mean"] == ddf[col].mean().compute()
            else:
                assert output[col]["avg"] == int(ddf[col].mean().compute())
        # Check cardinality for cat and label
        if col in columns_dict["cats"] + columns_dict["labels"]:
            assert output[col]["cardinality"] == ddf[col].nunique().compute()
        # Check std for cont
        if col in columns_dict["conts"]:
            assert output[col]["std"] == ddf[col].std().compute()

    # Stop Dask Cluster
    client.shutdown()
