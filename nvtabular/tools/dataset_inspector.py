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

import cudf
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
    """

    def __get_stats(self, ddf, ddf_dtypes, col, data, col_type, key_names):
        """
        Parameters
        -----------
        ddf : dask.dataframe.DataFrame
            Dask dataframe with the data
        ddf : dask.dataframe.DataFrame
            Dask dataframe with the correct dtypes
        col: string
            Col to process
        data: Dictionary
            Dictionary to store the output stats
        col_type: tring
            Column type (i.e cat, cont, label)
        key_names: Dictionary
            Dictionary with store dict names mapping
        """
        data[col] = {}
        # Get dtype
        data[col]["dtype"] = str(ddf_dtypes[col].dtype)

        # If string, chane string for its len
        if data[col]["dtype"] == "object":
            # data[col]["dtype"] = "string"
            data[col]["cardinality"] = ddf[col].nunique().compute()
            ddf[col] = ddf[col].map_partitions(lambda x: x.str.len())
            ddf[col].compute()

        # Get cardinality for cat and label
        elif col_type == "cat" or col_type == "label":
            data[col]["cardinality"] = ddf[col].nunique().compute()

        # If multihot, compute list content stats, and change list for its len
        if data[col]["dtype"] == "list":
            ddf._meta[col] = cudf.Series([np.array([0], dtype=ddf_dtypes[col].dtype.leaf_type)])[:0]
            # If list content is string
            if ddf_dtypes[col].dtype.leaf_type == "object":
                data[col]["multi_min"] = (
                    ddf[col].compute().list.leaves.applymap(lambda x: x.str.len()).min()
                )
                data[col]["multi_max"] = (
                    ddf[col].compute().list.leaves.applymap(lambda x: x.str.len()).max()
                )
                data[col]["multi_avg"] = (
                    ddf[col].compute().list.leaves.applymap(lambda x: x.str.len()).mean()
                )
            # If list content is a number
            else:
                data[col]["multi_min"] = ddf[col].compute().list.leaves.min()
                data[col]["multi_max"] = ddf[col].compute().list.leaves.max()
                data[col]["multi_avg"] = ddf[col].compute().list.leaves.mean()
            # Get list len for min/max/mean entry computation
            # TODO: Add when cudf support is implemented
            # https://github.com/rapidsai/cudf/issues/7157
            # ddf[col] = ddf[col].map_partitions(lambda x: x.list.len(),
            #                                    meta=(col, ddf_dtypes[col].dtype))
            # ddf[col].compute()

        # Get max/min/mean for all but label
        if col_type != "label":
            data[col][key_names["min"][col_type]] = ddf[col].min().compute()
            data[col][key_names["max"][col_type]] = ddf[col].max().compute()
            if col_type == "cont":
                data[col][key_names["mean"][col_type]] = ddf[col].mean().compute()
            else:
                data[col][key_names["mean"][col_type]] = int(ddf[col].mean().compute())

        # For conts get also std
        if col_type == "cont":
            data[col]["std"] = ddf[col].std().compute()

        # Get percentage of nan for all
        data[col]["per_nan"] = 100 * (1 - ddf[col].count().compute() / len(ddf[col]))

    def inspect(self, path, dataset_format, columns_dict, output_file):
        """
        Parameters
        -----------
        path: str, list of str, or <dask.dataframe|cudf|pd>.DataFrame
            Dataset path (or list of paths), or a DataFrame. If string,
            should specify a specific file or directory path. If this is a
            directory path, the directory structure must be flat (nested
            directories are not yet supported).
        dataset_format: string
            Dataset format (i.e parquet or csv)
        columns_dict: dictionary
            Dictionary indicating the diferent columns type
        output_file: string
            Filename to write the output statistics
        """
        # Get dataset columns
        cats = columns_dict["cats"]
        conts = columns_dict["conts"]
        labels = columns_dict["labels"]

        # Dictionary with json output key names
        key_names = {}
        key_names["min"] = {}
        key_names["min"]["cat"] = "min_entry_size"
        key_names["min"]["cont"] = "min_val"
        key_names["max"] = {}
        key_names["max"]["cat"] = "max_entry_size"
        key_names["max"]["cont"] = "max_val"
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

        # Compute first row to know correct dtypes
        ddf_dtypes = ddf.head(1)

        # Get continuous columns stats
        data["conts"] = {}
        for col in conts:
            self.__get_stats(ddf, ddf_dtypes, col, data["conts"], "cont", key_names)

        # Get categoricals columns stats
        data["cats"] = {}
        for col in cats:
            self.__get_stats(ddf, ddf_dtypes, col, data["cats"], "cat", key_names)

        # Get labels columns stats
        data["labels"] = {}
        for col in labels:
            self.__get_stats(ddf, ddf_dtypes, col, data["labels"], "label", key_names)

        # Write json file
        with fsspec.open(output_file, "w") as outfile:
            json.dump(data, outfile, cls=NpEncoder)

        # Stop Dask Cluster
        client.shutdown()
        cluster.close()
