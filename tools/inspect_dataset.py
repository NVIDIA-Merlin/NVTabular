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

import argparse
import json

import fsspec
import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular as nvt


# Class to help Json to serialize the data
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description=("Dataset Inspect Tool"))
    # Config file
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        help="Dataset columns type (Required)",
    )
    # Dataset path
    parser.add_argument(
        "-d",
        "--data_path",
        default="0",
        type=str,
        help="Input dataset path (Required)",
    )
    # Dataset format
    parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        type=str,
        help="Dataset format (Default 'parquet')",
    )
    # Output file name
    parser.add_argument(
        "-o",
        "--output_file",
        default="dataset_info.json",
        type=str,
        help="Output file name (Default 'dataset_info.json')",
    )
    args = parser.parse_args()
    return args


def get_stats(ddf, col, data, col_type):
    data[col] = {}

    # Get dtype and convert cat-stings and cat_mh-lists
    data[col]["dtype"] = str(ddf[col].dtype)
    if data[col]["dtype"] == "object":
        if col_type == "cat":
            data[col]["dtype"] = "string"
            ddf[col] = ddf[col].map_partitions(lambda x: x.str.len())
        elif col_type == "cat_mh":
            data[col]["dtype"] = "list"
            ddf[col] = ddf[col].map_partitions(lambda x: x.list.len())
        ddf[col].compute()

    # Get percentage of nan for all
    data[col]["nans_%"] = 100 * (1 - ddf[col].count().compute() / len(ddf[col]))

    # Get cardinality for cat and label
    data[col]["cardinality"] = ddf[col].nunique().compute()

    # Get max/min/mean for cat, cat_mh, and cont
    if col_type != "label":
        data[col]["min"] = ddf[col].min().compute()
        data[col]["max"] = ddf[col].max().compute()
        if col_type == "cont":
            data[col]["mean"] = ddf[col].mean().compute()
        else:
            data[col]["avg"] = int(ddf[col].mean().compute())

    # For conts get also std
    if col_type == "cont":
        data[col]["std"] = ddf[col].std().compute()


def main(args):
    # Get dataset columns
    with fsspec.open(args.config_file) as f:
        config = json.load(f)
    cats = config["cats"]
    cats_mh = config["cats_mh"]
    conts = config["conts"]
    labels = config["labels"]

    # Get dataset
    dataset = nvt.Dataset(args.data_path, engine=args.format)
    ddf = dataset.to_ddf()
    print(ddf.dtypes)

    # Create Dask Cluster
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)

    # Dictionary to store collected information
    data = {}
    # Store general info
    data["num_rows"] = ddf.shape[0].compute()
    data["cats"] = cats
    data["cats_mh"] = cats_mh
    data["conts"] = conts
    data["labels"] = labels

    # Get categoricals columns stats
    for col in cats:
        get_stats(ddf, col, data, "cat")

    # Get categoricals multihot columns stats
    for col in cats_mh:
        get_stats(ddf, col, data, "cat_mh")

    # Get continuous columns stats
    for col in conts:
        get_stats(ddf, col, data, "cont")

    # Get labels columns stats
    for col in conts:
        get_stats(ddf, col, data, "label")

    # Write json file
    with fsspec.open(args.output_file, "w") as outfile:
        json.dump(data, outfile, cls=NpEncoder)

    # Stop Dask Cluster
    client.shutdown()


if __name__ == "__main__":
    main(parse_args())
