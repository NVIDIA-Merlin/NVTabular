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
import cudf
import numpy as np
from cudf.utils.dtypes import is_list_dtype
from nvtx import annotate

from .stat_operator import StatOperator

from .operator import ColumnNames, Operator
import yaml

class DataStats(StatOperator):
    def __init__(self):
        super().__init__()
        self.output = {}

    @annotate("DataStats_fit", color="green", domain="nvt_python")
    def fit(self, columns: ColumnNames, ddf: dask_cudf.DataFrame):
        # Lists for columns values
        l_col_name = []
        l_col_type = []
        l_dtype = []
        l_cardinality = []
        l_min_val = []
        l_max_val = []
        l_std_val = []
        l_pernan_val = []
        # list for indexing
        l_cols = [l_col_name, l_col_type, l_dtype, l_cardinality, l_min_val, l_max_val, l_std_val, l_pernan_val]
        ddf_col_name = ["col_name", "col_type", "dtype", "cardinality", "min_val", "max_val", "std_val", "pernan_val"]

        # For each column, calculate the stats
        for col in columns:
            # Get dtype for all
            dtype = ddf[col].dtype
            # Identify column type (cont, cat, or cat_mh)
            if np.issubdtype(dtype, np.float):
                col_type = "cont"
            else:
                col_type = "cat"
                
            # Get cardinality for all but conts
            cardinality = ddf[col].nunique()

            # if string, replace string for their lengths for the rest of the computations
            if dtype == "object":
                ddf[col] = ddf[col].map_partitions(lambda x: x.str.len())
            # if list, replace lists by their length
            # This will fail right now, we are waiting for cudf to add support:
            # https://github.com/rapidsai/cudf/issues/7157
            #elif col_type == "cat_mh":
            #    ddf[col] = ddf[col].map_partitions(lambda x: x.list.len())

            # Get min,max, and mean for all
            min_val = ddf[col].min()
            max_val = ddf[col].max()
            mean_val = ddf[col].mean()

            # Get std only for conts
            std_val = ddf[col].std() if col_type == "cont" else 0

            # Get Percentage of NaNs for all
            pernan_val = 100 * (1 - ddf[col].count() / len(ddf[col]))

            # Create list for indexing
            l_temp = [col, col_type, str(dtype), cardinality, min_val, max_val, std_val, pernan_val]

            # Append stats to lists
            for i, l in enumerate(l_cols):
                l.append(l_temp[i])

        # Add column to dask_stats
        # Create cudf DataFrame to store column stats
        df = cudf.DataFrame()
        df["Init"] = [0] * len(columns)
        dask_stats = dask_cudf.from_cudf(df, npartitions=2)
        for i, l in enumerate(l_cols):
            _, name, meta, divisions = dask_stats. __getstate__()
            print(dask_cudf.Series(l, name, cudf.Series(), divisions))
            dask_stats[ddf_col_name[i]] = dask_cudf.Series(l, name, cudf.Series(), divisions)
        dask_stats = dask_stats.drop("Init", axis=1)

        return dask_stats

    def fit_finalize(self, dask_stats):
        self.output = dask_stats.to_pandas().set_index("col_name").T.to_dict()
        print(self.output)

    def save(self):
        return self.output

    def load(self, data):
        self.output = data

    def clear(self):
        self.output = {}

    #transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
