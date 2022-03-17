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

from .operator import ColumnSelector, Operator
from .stat_operator import StatOperator


class FillMissing(Operator):
    """
    This operation replaces missing values with a constant pre-defined value

    Example usage::

        # Use FillMissing to define a workflow for continuous columns and specify the fill value
        # Default is 0
        cont_features = ['cont1', 'cont2', 'cont3'] >> ops.FillMissing() >> ...
        processor = nvtabular.Workflow(cont_features)

    Parameters
    -----------
    fill_val : float, default 0
        The constant value to replace missing values with.
    add_binary_cols : boolean, default False
        When True, adds binary columns that indicate whether cells in each column were filled
    """

    def __init__(self, fill_val=0, add_binary_cols=False):
        super().__init__()
        self.fill_val = fill_val
        self.add_binary_cols = add_binary_cols
        self._inference_transform = None

    @annotate("FillMissing_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        if self.add_binary_cols:
            for col in col_selector.names:
                df[f"{col}_filled"] = df[col].isna()
                df[col] = df[col].fillna(self.fill_val)
        else:
            df[col_selector.names] = df[col_selector.names].fillna(self.fill_val)

        return df

    def inference_initialize(self, col_selector, inference_config):
        """load up extra configuration about this op."""
        if self.add_binary_cols:
            return None
        import nvtabular_cpp

        return nvtabular_cpp.inference.FillTransform(self)

    def column_mapping(self, col_selector):
        column_mapping = super().column_mapping(col_selector)
        for col_name in col_selector.names:
            if self.add_binary_cols:
                column_mapping[f"{col_name}_filled"] = [col_name]
        return column_mapping

    def _compute_dtype(self, col_schema, input_schema):
        col_schema = super()._compute_dtype(col_schema, input_schema)
        if col_schema.name.endswith("_filled"):
            col_schema = col_schema.with_dtype(np.bool)
        return col_schema

    transform.__doc__ = Operator.transform.__doc__


class FillMedian(StatOperator):
    """
    This operation replaces missing values with the median value for the column.

    Example usage::

        # Use FillMedian in a workflow for continuous columns
        cont_features = ['cont1', 'cont2', 'cont3'] >> ops.FillMedian()
        processor = nvtabular.Workflow(cont_features)

    Parameters
    -----------
    add_binary_cols : boolean, default False
        When True, adds binary columns that indicate whether cells in each column were filled
    """

    def __init__(self, add_binary_cols=False):
        super().__init__()
        self.add_binary_cols = add_binary_cols
        self.medians = {}

    @annotate("FillMedian_transform", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        if not self.medians:
            raise RuntimeError("need to call 'fit' before running transform")

        for col in col_selector.names:
            if self.add_binary_cols:
                df[f"{col}_filled"] = df[col].isna()
            df[col] = df[col].fillna(self.medians[col])
        return df

    @annotate("FillMedian_fit", color="green", domain="nvt_python")
    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame):
        # TODO: Use `method="tidigest"` when crick supports device
        dask_stats = ddf[col_selector.names].quantile(q=0.5, method="dask")
        return dask_stats

    @annotate("FillMedian_finalize", color="green", domain="nvt_python")
    def fit_finalize(self, dask_stats):
        index = dask_stats.index
        vals = index.values_host if hasattr(index, "values_host") else index.values
        for col in vals:
            self.medians[col] = float(dask_stats[col])

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__

    def clear(self):
        self.medians = {}

    def column_mapping(self, col_selector):
        column_mapping = super().column_mapping(col_selector)
        for col_name in col_selector.names:
            if self.add_binary_cols:
                column_mapping[f"{col_name}_filled"] = [col_name]
        return column_mapping

    def _compute_dtype(self, col_schema, input_schema):
        col_schema = super()._compute_dtype(col_schema, input_schema)
        if col_schema.name.endswith("_filled"):
            col_schema = col_schema.with_dtype(np.bool)
        return col_schema
