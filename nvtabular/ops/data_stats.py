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
import cudf
import dask_cudf
import numpy as np
from nvtx import annotate

from .operator import ColumnNames, Operator
from .stat_operator import StatOperator


class DataStats(StatOperator):
    def __init__(self):
        super().__init__()
        self.col_names = []
        self.col_types = []
        self.col_dtypes = []
        self.output = {}

    def transform(self, columns: ColumnNames, gdf: cudf.DataFrame) -> cudf.DataFrame:
        return gdf

    @annotate("DataStats_fit", color="green", domain="nvt_python")
    def fit(self, columns: ColumnNames, ddf: dask_cudf.DataFrame):
        dask_stats = {}

        ddf_dtypes = ddf.head(1)

        # For each column, calculate the stats
        for col in columns:
            dask_stats[col] = {}
            self.col_names.append(col)
            # Get dtype for all
            dtype = ddf_dtypes[col].dtype
            self.col_dtypes.append(dtype)

            # Identify column type
            if np.issubdtype(dtype, np.floating):
                col_type = "conts"
            else:
                col_type = "cats"
            self.col_types.append(col_type)

            # Get cardinality for cats
            if col_type == "cats":
                dask_stats[col]["cardinality"] = ddf[col].nunique()

            # if string, replace string for their lengths for the rest of the computations
            if dtype == "object":
                ddf[col] = ddf[col].map_partitions(lambda x: x.str.len(), meta=("x", int))
            # Add list support when cudf supports it:
            # https://github.com/rapidsai/cudf/issues/7157
            # elif col_type == "cat_mh":
            #    ddf[col] = ddf[col].map_partitions(lambda x: x.list.len())

            # Get min,max, and mean
            dask_stats[col]["min"] = ddf[col].min()
            dask_stats[col]["max"] = ddf[col].max()
            dask_stats[col]["mean"] = ddf[col].mean()

            # Get std only for conts
            if col_type == "conts":
                dask_stats[col]["std"] = ddf[col].std()

            # Get Percentage of NaNs for all
            dask_stats[col]["per_nan"] = 100 * (1 - ddf[col].count() / len(ddf[col]))

        return dask_stats

    def fit_finalize(self, dask_stats):
        for i, col in enumerate(self.col_names):
            # Add dtype
            dask_stats[col]["dtype"] = str(self.col_dtypes[i])
            # Cast types for yaml
            if isinstance(dask_stats[col]["mean"], np.floating):
                dask_stats[col]["mean"] = dask_stats[col]["mean"].item()
            if isinstance(dask_stats[col]["per_nan"], np.floating):
                dask_stats[col]["per_nan"] = dask_stats[col]["per_nan"].item()
            if self.col_types[i] == "conts":
                if isinstance(dask_stats[col]["std"], np.floating):
                    dask_stats[col]["std"] = dask_stats[col]["std"].item()
            else:
                if isinstance(dask_stats[col]["cardinality"], np.integer):
                    dask_stats[col]["cardinality"] = dask_stats[col]["cardinality"].item()
        self.output = dask_stats

    def clear(self):
        self.output = {}

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
