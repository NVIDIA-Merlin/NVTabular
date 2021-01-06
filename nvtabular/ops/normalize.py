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
import cudf
from nvtx import annotate

from .moments import _custom_moments
from .stat_operator import StatOperator


class Normalize(StatOperator):
    """
    Standardizing the features around 0 with a standard deviation
    of 1 is a common technique to compare measurements that have
    different units. This operation can be added to the workflow
    to standardize the features.

    It performs Normalization using the mean std method.

    Example usage::

        # Use Normalize to define a NVTabular workflow
        cont_features = CONTINUOUS_COLUMNS >> ops.Normalize()
        processor = nvtabular.Workflow(cont_features)

    Parameters
    ----------

    """

    def __init__(self):
        super().__init__()
        self.means = {}
        self.stds = {}

    @annotate("Normalize_fit", color="green", domain="nvt_python")
    def fit(self, columns, ddf):
        return _custom_moments(ddf[columns])

    def fit_finalize(self, dask_stats):
        for col in dask_stats.index:
            self.means[col] = float(dask_stats["mean"].loc[col])
            self.stds[col] = float(dask_stats["std"].loc[col])

    @annotate("Normalize_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns, gdf: cudf.DataFrame):
        new_gdf = cudf.DataFrame()
        for name in columns:
            if self.stds[name] > 0:
                new_gdf[name] = (gdf[name] - self.means[name]) / (self.stds[name])
                new_gdf[name] = new_gdf[name].astype("float32")
        return new_gdf

    def save(self):
        return {"means": self.means, "stds": self.stds}

    def load(self, data):
        self.means = data["means"]
        self.stds = data["stds"]

    def clear(self):
        self.means = {}
        self.stds = {}


class NormalizeMinMax(StatOperator):
    """
    Standardizing the features around 0 with a standard deviation
    of 1 is a common technique to compare measurements that have
    different units. This operation can be added to the workflow
    to standardize the features.

    It performs Normalization using the min max method.

    Example usage::

        # Use NormalizeMinMax to define a NVTabular workflow
        cont_features = CONTINUOUS_COLUMNS >> ops.NormalizeMinMax()
        processor = nvtabular.Workflow(cont_features)

    Parameters
    ----------

    """

    def __init__(self):
        self.mins = {}
        self.maxs = {}

    @annotate("NormalizeMinMax_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns, gdf: cudf.DataFrame):
        new_gdf = cudf.DataFrame()
        for name in columns:
            dif = self.maxs[name] - self.mins[name]
            if dif > 0:
                new_gdf[name] = (gdf[name] - self.mins[name]) / dif
            elif dif == 0:
                new_gdf[name] = gdf[name] / (2 * gdf[name])
            new_gdf[name] = new_gdf[name].astype("float32")
        return new_gdf

    @annotate("MinMax_op", color="green", domain="nvt_python")
    def fit(self, columns, ddf):
        return {
            "mins": ddf[columns].min(),
            "maxs": ddf[columns].max(),
        }

    @annotate("MinMax_finalize", color="green", domain="nvt_python")
    def fit_finalize(self, dask_stats):
        for col in dask_stats["mins"].index.values_host:
            self.mins[col] = dask_stats["mins"][col]
            self.maxs[col] = dask_stats["maxs"][col]

    def save(self):
        return {"mins": self.mins, "maxs": self.maxs}

    def load(self, data):
        self.mins = data["mins"]
        self.maxs = data["maxs"]

    def clear(self):
        self.mins = {}
        self.maxs = {}
