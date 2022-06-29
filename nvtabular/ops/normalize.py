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
import dask.dataframe as dd
import numpy

from merlin.core.dispatch import (
    DataFrameType,
    annotate,
    encode_list_column,
    flatten_list_column_values,
    is_list_dtype,
)
from merlin.dag import Supports
from merlin.schema import Tags

from .moments import _custom_moments
from .operator import ColumnSelector, Operator
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
    -----------
    out_dtype : str
        Specifies the data type for the output columns. The default value is `numpy.float64` if
        not set here
    """

    def __init__(self, out_dtype=None):
        super().__init__()
        self.means = {}
        self.stds = {}
        self.out_dtype = out_dtype

    @annotate("Normalize_fit", color="green", domain="nvt_python")
    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame):
        return _custom_moments(ddf[col_selector.names])

    def fit_finalize(self, dask_stats):
        for col in dask_stats.index:
            self.means[col] = float(dask_stats["mean"].loc[col])
            self.stds[col] = float(dask_stats["std"].loc[col])

    @annotate("Normalize_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        new_df = type(df)()
        for name in col_selector.names:
            values = df[name]
            list_col = is_list_dtype(values)
            if list_col:
                values = flatten_list_column_values(values)

            if self.stds[name] > 0:
                values = (values - self.means[name]) / (self.stds[name])
            else:
                values = values - self.means[name]

            values = values.astype(self.output_dtype)

            if list_col:
                values = encode_list_column(df[name], values)

            new_df[name] = values
        return new_df

    @property
    def supports(self):
        return (
            Supports.CPU_DICT_ARRAY
            | Supports.GPU_DICT_ARRAY
            | Supports.CPU_DATAFRAME
            | Supports.GPU_DATAFRAME
        )

    def clear(self):
        self.means = {}
        self.stds = {}

    @property
    def output_tags(self):
        return [Tags.CONTINUOUS]

    @property
    def output_dtype(self):
        return self.out_dtype or numpy.float64

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

    Parameters
    -----------
    out_dtype : str, default is float64
        dtype of output columns.
    """

    def __init__(self, out_dtype=None):
        super().__init__()
        self.mins = {}
        self.maxs = {}
        self.out_dtype = out_dtype

    @annotate("NormalizeMinMax_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType):
        # TODO: should we clip values if they are out of bounds (below 0 or 1)
        # (could happen in validation dataset if datapoint)
        new_df = type(df)()
        for name in col_selector.names:
            dif = self.maxs[name] - self.mins[name]
            if dif > 0:
                new_df[name] = (df[name] - self.mins[name]) / dif
            elif dif == 0:
                new_df[name] = df[name] / (2 * df[name])
            new_df[name] = new_df[name].astype(self.output_dtype)
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    @annotate("NormalizeMinMax_fit", color="green", domain="nvt_python")
    def fit(self, col_selector: ColumnSelector, ddf):
        return {
            "mins": ddf[col_selector.names].min(),
            "maxs": ddf[col_selector.names].max(),
        }

    @annotate("NormalizeMinMax_finalize", color="green", domain="nvt_python")
    def fit_finalize(self, dask_stats):
        index = dask_stats["mins"].index
        cols = index.values_host if hasattr(index, "values_host") else index.values
        for col in cols:
            self.mins[col] = dask_stats["mins"][col]
            self.maxs[col] = dask_stats["maxs"][col]

    def clear(self):
        self.mins = {}
        self.maxs = {}

    @property
    def supports(self):
        return (
            Supports.CPU_DICT_ARRAY
            | Supports.GPU_DICT_ARRAY
            | Supports.CPU_DATAFRAME
            | Supports.GPU_DATAFRAME
        )

    @property
    def output_tags(self):
        return [Tags.CONTINUOUS]

    @property
    def output_dtype(self):
        return self.out_dtype or numpy.float64

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
