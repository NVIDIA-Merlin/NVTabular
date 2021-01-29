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
import dask_cudf
from nvtx import annotate

from .moments import _custom_moments
from .operator import ColumnNames, Operator
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
    def fit(self, columns: ColumnNames, ddf: dask_cudf.DataFrame):
        return _custom_moments(ddf[columns])

    def fit_finalize(self, dask_stats):
        for col in dask_stats.index:
            self.means[col] = float(dask_stats["mean"].loc[col])
            self.stds[col] = float(dask_stats["std"].loc[col])

    @annotate("Normalize_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, gdf: cudf.DataFrame) -> cudf.DataFrame:
        new_gdf = cudf.DataFrame()
        for name in columns:
            if self.stds[name] > 0:
                new_gdf[name] = (gdf[name] - self.means[name]) / (self.stds[name])
                new_gdf[name] = new_gdf[name].astype("float32")
        return new_gdf

    def clear(self):
        self.means = {}
        self.stds = {}

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__


class NormalizeMinMax(StatOperator):
    """
    This operator standardizes continuous features such that they are between 0 and 1.

    Example usage::
        # Use NormalizeMinMax to define a NVTabular workflow
        cont_features = CONTINUOUS_COLUMNS >> ops.NormalizeMinMax()
        processor = nvtabular.Workflow(cont_features)
    """

    def __init__(self):
        self.mins = {}
        self.maxs = {}

    @annotate("NormalizeMinMax_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns, gdf: cudf.DataFrame):
        # TODO: should we clip values if they are out of bounds (below 0 or 1)
        # (could happen in validation dataset if datapoint)
        new_gdf = cudf.DataFrame()
        for name in columns:
            dif = self.maxs[name] - self.mins[name]
            if dif > 0:
                new_gdf[name] = (gdf[name] - self.mins[name]) / dif
            elif dif == 0:
                new_gdf[name] = gdf[name] / (2 * gdf[name])
            new_gdf[name] = new_gdf[name].astype("float32")
        return new_gdf

    transform.__doc__ = Operator.transform.__doc__

    @annotate("NormalizeMinMax_fit", color="green", domain="nvt_python")
    def fit(self, columns, ddf):
        return {
            "mins": ddf[columns].min(),
            "maxs": ddf[columns].max(),
        }

    @annotate("NormalizeMinMax_finalize", color="green", domain="nvt_python")
    def fit_finalize(self, dask_stats):
        for col in dask_stats["mins"].index.values_host:
            self.mins[col] = dask_stats["mins"][col]
            self.maxs[col] = dask_stats["maxs"][col]

    def clear(self):
        self.mins = {}
        self.maxs = {}

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
