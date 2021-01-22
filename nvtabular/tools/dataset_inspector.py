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
import json
from contextlib import contextmanager

import cudf
import fsspec
import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from nvtabular.io import Dataset
from nvtabular.ops import DataStats
from nvtabular.workflow import Workflow


# Context Manager for creating Dask Cluster
def client():
    client = Client(LocalCluster())
    yield client
    client.close()


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


class DatasetInspector:
    """
    Analyzes an existing dataset to extract its statistics.
    """

    def inspect(self, path, dataset_format, columns_dict, output_file):
        """
        Parameters
        -----------
        path: str, list of str, or <dask.dataframe|cudf|pd>.DataFrame
            Dataset path (or list of paths), or a DataFrame. If string,
            should specify a specific file or directory path. If this is a
            directory path, the directory structure must be flat (nested
            directories are not yet supported).
        dataset_format: str
            Dataset format (i.e parquet or csv)
        columns_dict: dict
            Dictionary indicating the diferent columns type
        output_file: str
            Filename to write the output statistics
        """
        # Get dataset columns
        cats = columns_dict["cats"]
        conts = columns_dict["conts"]
        labels = columns_dict["labels"]
        all_cols = cats + conts + labels

        # Dictionary with json output key names
        key_names = {}
        key_names["min"] = {}
        key_names["min"]["cats"] = "min_entry_size"
        key_names["min"]["conts"] = "min_val"
        key_names["max"] = {}
        key_names["max"]["cats"] = "max_entry_size"
        key_names["max"]["conts"] = "max_val"
        key_names["mean"] = {}
        key_names["mean"]["cats"] = "avg_entry_size"
        key_names["mean"]["conts"] = "mean"

        # Create Dataset, Workflow, and get Stats
        dataset = Dataset(path, engine=dataset_format)
        features = all_cols >> DataStats()
        workflow = Workflow(features, client=client())
        workflow.fit(dataset)
        # Save stats in a file and read them back
        stats_file = "stats_output.yaml"
        workflow.save_stats(stats_file)
        output = yaml.safe_load(open(stats_file))
        
        
        # Dictionary to store collected information
        data = {}
        # Store num_rows
        data["num_rows"] = dataset.num_rows
        # Store cols
        for col_type in ["conts", "cats", "labels"]:
            for col in all_cols:
                data[col_type][col]["dtype"] = output[col][0]
                data[col_type][col]["cardinality"] = output[col][1]
                if col_type != "label":
                    data[col_type][col][key_names["min"][col_type]] = output[col][2]
                    data[col_type][col][key_names["max"][col_type]] = output[col][3]
                    data[col_type][col][key_names["mean"][col_type]] = output[col][4]
                if col_type == "cont":
                    data[col_type][col]["std"] = output[col][5]
                data[col_type][col]["per_nan"] = output[col][6]

        # Write json file
        with fsspec.open(output_file, "w") as outfile:
            json.dump(data, outfile, cls=NpEncoder)
