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
import fsspec
import json
import nvtabular as nvt
import numpy as np
from functools import singledispatch

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
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
        help='Dataset columns type (Required)',
    )
    # Dataset path
    parser.add_argument(
        "-d",
        "--data_path",
        default="0",
        type=str,
        help='Input dataset path (Required)',
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

def get_all_stats(ddf, col, data):
    data[col] = {}
    # Get dtype
    data[col]['dtype'] = str(ddf[col].dtype) 
    if data[col]['dtype'] == "object":
        data[col]['dtype'] = "string"
    data[col]['nans_%'] = 100 * (1 - ddf[col].count().compute() / len(ddf[col]))

def get_cats_stats(ddf, col, data):
    data[col]['cardinality'] = ddf[col].nunique().compute()
    
def get_conts_stats(ddf, col, data):
    data[col]['min'] = ddf[col].min().compute()
    data[col]['max'] = ddf[col].max().compute()
    data[col]['mean'] = ddf[col].mean().compute()
    data[col]['std'] = ddf[col].std().compute()

def get_labels_stats(ddf, col, data):
    data[col]['cardinality'] = ddf[col].nunique().compute()

def main(args):
    # Get dataset columns
    with fsspec.open(args.config_file) as f:
        config = json.load(f)
    cats = config['cats']
    conts = config['conts']
    labels = config['labels']
    columns = cats+conts+labels

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
    # data['num_rows'] = dataset.num_rows
    data['cats'] = cats
    data['conts'] = conts
    data['labels'] = labels

    # Get continuous columnd stats
    for col in cats:
        get_all_stats(ddf, col, data)
        get_cats_stats(ddf, col, data)

    # Get continuous columnd stats
    for col in conts:
        get_all_stats(ddf, col, data)
        get_conts_stats(ddf, col, data)

    # Get labels columns stats
    for col in conts:
        get_all_stats(ddf, col, data)
        get_labels_stats(ddf, col, data)
    
    print(data)

    # Write json file
    with fsspec.open(args.output_file, 'w') as outfile:
        json.dump(data, outfile, cls=NpEncoder)

if __name__ == "__main__":
    main(parse_args())