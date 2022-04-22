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

import json

import fsspec
import numpy as np

from merlin.core.utils import set_client_deprecated
from merlin.dag import ColumnSelector
from nvtabular.ops import DataStats
from nvtabular.workflow import Workflow


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

    def __init__(self, client=None):
        # Deprecate `client`
        if client:
            set_client_deprecated(client, "DatasetInspector")

    def inspect(self, dataset, columns_dict, output_file):
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
            Dictionary indicating the different columns type
        output_file: str
            Filename to write the output statistics
        """

        # Get dataset columns
        cats = columns_dict["cats"]
        conts = columns_dict["conts"]
        labels = columns_dict["labels"]

        # Create Dataset, Workflow, and get Stats
        stats = DataStats()
        features = ColumnSelector(cats + conts + labels) >> stats
        workflow = Workflow(features)
        workflow.fit(dataset)

        # get statistics from the datastats op
        output = stats.output

        # Dictionary to store collected information
        data = {}
        # Store num_rows
        data["num_rows"] = dataset.num_rows
        # Store cols
        for col_type in ["conts", "cats", "labels"]:
            data[col_type] = {}
            for col in columns_dict[col_type]:
                data[col_type][col] = {}
                data[col_type][col]["dtype"] = output[col]["dtype"]

                if col_type != "conts":
                    data[col_type][col]["cardinality"] = output[col]["cardinality"]

                if col_type == "cats":
                    data[col_type][col]["min_entry_size"] = output[col]["min"]
                    data[col_type][col]["max_entry_size"] = output[col]["max"]
                    data[col_type][col]["avg_entry_size"] = output[col]["mean"]
                elif col_type == "conts":
                    data[col_type][col]["min_val"] = output[col]["min"]
                    data[col_type][col]["max_val"] = output[col]["max"]
                    data[col_type][col]["mean"] = output[col]["mean"]
                    data[col_type][col]["std"] = output[col]["std"]

                data[col_type][col]["per_nan"] = output[col]["per_nan"]

        # Write json file
        with fsspec.open(output_file, "w") as outfile:
            json.dump(data, outfile, cls=NpEncoder)
