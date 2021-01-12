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

import fsspec
import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from nvtabular.io import Dataset

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

    Parameters
    -----------
    path_or_source : str, list of str, or <dask.dataframe|cudf|pd>.DataFrame
        Dataset path (or list of paths), or a DataFrame. If string,
        should specify a specific file or directory path. If this is a
        directory path, the directory structure must be flat (nested
        directories are not yet supported).
    columns_dict: dictionary
        Dictionary indicating the diferent columns type
    """

    def __get_stats(self, ddf, col, data, col_type, key_names):
        data[col] = {}
        # Get dtype and convert cat-strings and cat_mh-lists
        # TODO: Fix dask-cudf _meta
        data[col]["dtype"] = str(ddf[col].dtype)
        if data[col]["dtype"] == "object":
            if col_type == "cat":
                data[col]["dtype"] = "string"
                ddf[col] = ddf[col].map_partitions(lambda x: x.str.len())
            elif col_type == "cat_mh":
                data[col]["dtype"] = "list"
                # For multihot columns, calculate min/max/mean for the content of the lists
                if ddf[col].dtype is cudf.core.dtypes.ListDtype:
                # For string
                if ddf[col].dtype.leaf_type == "string":
                    data[col]["multi_min"] = ddf[col].compute().list.leaves.applymap(lambda x: x.str.len()).min()
                    data[col]["multi_max"] = ddf[col].compute().list.leaves.applymap(lambda x: x.str.len()).max()
                    data[col]["multi_avg"] = ddf[col].compute().list.leaves.applymap(lambda x: x.str.len()).mean()
                # For int/float
                else:
                    data[col]["multi_min"] = ddf[col].compute().list.leaves.min()
                    data[col]["multi_max"] = ddf[col].compute().list.leaves.max()
                    data[col]["multi_avg"] = ddf[col].compute().list.leaves.mean()
                # Get list size for min/max/mean entry computation
                ddf[col] = ddf[col].map_partitions(lambda x: x.list.len())
                ddf[col].compute()

        # Get max/min/mean for all but label
        if col_type != "label":
            data[col][key_names["min"][col_type]] = ddf[col].min().compute()
            data[col][key_names["max"][col_type]] = ddf[col].max().compute()
            data[col][key_names["mean"][col_type]] = ddf[col].mean().compute()
            
        # Get cardinality for cat and label
        if col_type == "cat" or col_type == "label":
            data[col]["cardinality"] = ddf[col].nunique().compute()

        # For conts get also std
        if col_type == "cont":
            data[col]["standard deviation"] = ddf[col].std().compute()

        # Get percentage of nan for all
        data[col]["nans_%"] = 100 * (1 - ddf[col].count().compute() / len(ddf[col]))

    def inspect(self, path, dataset_format, columns_dict, output_file):
        # Get dataset columns
        cats = columns_dict["cats"]
        cats_mh = columns_dict["cats_mh"]
        conts = columns_dict["conts"]
        labels = columns_dict["labels"]

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

        # Get dataset
        dataset = Dataset(path, engine=dataset_format)
        ddf = dataset.to_ddf()

        # Create Dask Cluster
        cluster = LocalCUDACluster()
        client = Client(cluster)

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
            self.__get_stats(ddf, col, data, "cat", key_names)

        # Get categoricals multihot columns stats
        for col in cats_mh:
            self.__get_stats(ddf, col, data, "cat_mh", key_names)

        # Get continuous columns stats
        for col in conts:
            self.__get_stats(ddf, col, data, "cont", key_names)

        # Get labels columns stats
        for col in labels:
            self.__get_stats(ddf, col, data, "label", key_names)

        # Write json file
        with fsspec.open(output_file, "w") as outfile:
            json.dump(data, outfile, cls=NpEncoder)

        # Stop Dask Cluster
        client.shutdown()
