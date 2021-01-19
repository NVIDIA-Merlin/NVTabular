import glob
import json
import os

import fsspec
import pytest
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular.tools.dataset_inspector as datains
from nvtabular.io import Dataset


@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_inspect(tmpdir, datasets, engine):
    # Dataset
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    output_file = tmpdir + "/dataset_info.json"

    # Dataset columns type config
    columns_dict = {}
    columns_dict["cats"] = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    columns_dict["conts"] = ["x", "y"]
    columns_dict["labels"] = ["label"]
    all_cols = columns_dict["cats"] + columns_dict["conts"] + columns_dict["labels"]

    # Create inspector and inspect
    a = datains.DatasetInspector()
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

    # Dictionary with json output key names
    key_names = {}
    key_names["min"] = {}
    key_names["min"]["cat"] = "min_entry_size"
    key_names["min"]["cont"] = "min_value"
    key_names["max"] = {}
    key_names["max"]["cat"] = "max_entry_size"
    key_names["max"]["cont"] = "max_value"
    key_names["mean"] = {}
    key_names["mean"]["cat"] = "avg_entry_size"
    key_names["mean"]["cont"] = "mean"
    # Correct dtypes
    ddf_dtypes = ddf.head(1)

    # Check output
    for col in all_cols:
        # Check dtype for all
        assert output[col]["dtype"] == str(ddf_dtypes[col].dtype)
        # Get string len for stats computation
        if output[col]["dtype"] == "object":
            ddf[col] = ddf[col].map_partitions(lambda x: x.str.len())
            ddf.compute()
        # Check lists stats
        elif output[col]["dtype"] == "list":
            if ddf_dtypes[col].dtype.leaf_type == "string":
                output[col]["multi_min"] == ddf[col].compute().list.leaves.applymap(
                    lambda x: x.str.len()
                ).min()
                output[col]["multi_max"] == ddf[col].compute().list.leaves.applymap(
                    lambda x: x.str.len()
                ).max()
                output[col]["multi_avg"] == ddf[col].compute().list.leaves.applymap(
                    lambda x: x.str.len()
                ).mean()
            else:
                output[col]["multi_min"] == ddf[col].compute().list.leaves.min()
                output[col]["multi_max"] == ddf[col].compute().list.leaves.max()
                output[col]["multi_avg"] == ddf[col].compute().list.leaves.mean()
            # Get list len for stats computation
            ddf[col] = ddf[col].map_partitions(lambda x: len(x), meta=(col, ddf_dtypes[col].dtype))
            ddf[col].compute()

        # Check percentage of nan for all
        assert output[col]["nans_%"] == (100 * (1 - ddf[col].count().compute() / len(ddf[col])))

        # Check max/min/mean for all but label
        if col not in columns_dict["labels"]:
            col_type = "cont" if col in columns_dict["conts"] else "cat"
            assert output[col][key_names["min"][col_type]] == ddf[col].min().compute()
            assert output[col][key_names["max"][col_type]] == ddf[col].max().compute()
            assert output[col][key_names["mean"][col_type]] == ddf[col].mean().compute()

        # Check cardinality for cat and label
        if col in columns_dict["cats"] + columns_dict["labels"]:
            assert output[col]["cardinality"] == ddf[col].nunique().compute()

        # Check std for cont
        if col in columns_dict["conts"]:
            assert output[col]["standard deviation"] == ddf[col].std().compute()

    # Stop Dask Cluster
    client.shutdown()
    cluster.close()
