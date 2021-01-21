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
import dask_cudf
import numpy as np
from cudf.utils.dtypes import is_list_dtype

from .stat_operator import StatOperator


class DataStats(StatOperator):
    def __init__(self):
        super().__init__()
        self.output = {}

    @annotate("DataStats_fit", color="green", domain="nvt_python")
    def fit(self, columns: ColumnNames, ddf: dask_cudf.DataFrame):
        # Create dask DataFrame to store column stats
        dask_stats = dask_cudf.DataFrame()

        # For each column, calculate the stats
        for col in columns:
            # Get dtype for all
            dtype = ddf[col].dtype
            # Identify column type (cont, cat, or cat_mh)
            if is_list_dtype(dtype):
                col_type = "cat_mh"
                # Get flat list leaves (How to avoid the compute?)
                if ddf_dtypes[col].dtype.leaf_type == "object":
                    flatten = ddf[col].list.leaves.applymap(lambda x: x.str.len())
                else:
                    flatten = ddf[col].list.leaves
            elif np.issubdtype(dtype, np.float):
                col_type = "cont"
            else:
                col_type = "cat"

            # Get cardinality for all but conts
            cardinality = None
            if col_type == "cat_mh":
                cardinality = flatten.nunique()
            elif col_type == "cat":
                cardinality = ddf[col].nunique()

            # if string, replace string for their lengths for the rest of the computations
            if dtype == "object":
                ddf[col] = ddf[col].map_partitions(lambda x: x.str.len())
            # if list, replace lists by their length
            # This will fail right now, we are waiting for cudf to add support:
            # https://github.com/rapidsai/cudf/issues/7157
            elif col_type == "cat_mh":
                ddf[col] = ddf[col].map_partitions(lambda x: x.list.len())

            # Get min,max, and mean for all
            min_val = ddf[col].min()
            max_val = ddf[col].max()
            mean_val = ddf[col].mean()

            # Get std only for conts
            std_val = ddf[col].std() if col_type == "cont" else None

            # Get Percentage of NaNs for all
            pernan_val = 100 * (1 - ddf[col].count() / len(ddf[col]))

            # Get multi stats for lists
            multi_min = flatten.min() if col_type == "cat_mh" else None
            multi_max = flatten.max() if col_type == "cat_mh" else None
            multi_mean = flatten.min() if col_type == "cat_mh" else None

            # Add column to dask_stats
            dask_stats[col] = [
                col_type,
                dtype,
                cardinality,
                min_val,
                max_val,
                std_val,
                pernan_val,
                multi_min,
                multi_max,
                multi_mean,
            ]
        return dask_stats

    def fit_finalize(self, dask_stats):
        for col in dask_stats:
            output[col] = dask_stats[col].to_list()

    def load(self, data):
        self.output = data

    def clear(self):
        self.output = {}

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
