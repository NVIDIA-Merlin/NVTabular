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
import numpy as np

from merlin.core.dispatch import DataFrameType, annotate
from merlin.schema import Schema

from .operator import ColumnSelector, Operator
from .stat_operator import StatOperator

_INT_DTYPES = [np.int8, np.int16, np.int32, np.int64]


class ReduceDtypeSize(StatOperator):
    """
    ReduceDtypeSize changes the dtypes of numeric columns. For integer columns
    this will choose a dtype such that the minimum and maximum values in the
    column will fit. For float columns this will cast to a float32.
    """

    def __init__(self, float_dtype=np.float32):
        super().__init__()
        self.float_dtype = float_dtype
        self.ranges = {}
        self.dtypes = {}

    @annotate("reduce_dtype_size_fit", color="green", domain="nvt_python")
    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame):
        return {col: (ddf[col].min(), ddf[col].max()) for col in col_selector.names}

    def fit_finalize(self, dask_stats):
        self.ranges = dask_stats

    def clear(self):
        self.dtypes = {}
        self.ranges = {}

    @annotate("reduce_dtype_size_transform", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        for col, dtype in self.dtypes.items():
            df[col] = df[col].astype(dtype)
        return df

    def compute_output_schema(self, input_schema, selector, prev_output_schema=None):
        if not self.ranges:
            return input_schema

        output_columns = []
        for column, (min_value, max_value) in self.ranges.items():
            column = input_schema[column]

            dtype = column.dtype
            if np.issubdtype(column.dtype, np.integer):
                for possible_dtype in _INT_DTYPES:
                    dtype_range = np.iinfo(possible_dtype)
                    if min_value >= dtype_range.min and max_value <= dtype_range.max:
                        dtype = possible_dtype
                        break

            elif np.issubdtype(column.dtype, np.float):
                dtype = self.float_dtype

            output_columns.append(column.with_dtype(dtype))

        self.dtypes = {column.name: column.dtype for column in output_columns}
        return Schema(output_columns)

    transform.__doc__ = Operator.transform.__doc__
    compute_output_schema.__doc__ = Operator.compute_output_schema.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__
    clear.__doc__ = StatOperator.clear.__doc__
