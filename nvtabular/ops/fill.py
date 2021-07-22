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

from nvtabular.dispatch import DataFrameType, annotate

from .operator import ColumnNames, Operator
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
    def transform(self, columns, df: DataFrameType) -> DataFrameType:
        if self.add_binary_cols:
            for col in columns:
                df[f"{col}_filled"] = df[col].isna()
                df[col] = df[col].fillna(self.fill_val)
        else:
            df[columns] = df[columns].fillna(self.fill_val)

        return df

    def inference_initialize(self, columns, inference_config):
        """ load up extra configuration about this op.  """
        if self.add_binary_cols:
            return None
        import nvtabular_cpp

        return nvtabular_cpp.inference.FillTransform(self)

    transform.__doc__ = Operator.transform.__doc__

    def output_column_names(self, columns: ColumnNames) -> ColumnNames:
        output_cols = columns[:]
        if self.add_binary_cols:
            output_cols.extend([f"{col}_filled" for col in columns])
        return output_cols


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
    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        if not self.medians:
            raise RuntimeError("need to call 'fit' before running transform")

        for col in columns:
            if self.add_binary_cols:
                df[f"{col}_filled"] = df[col].isna()
            df[col] = df[col].fillna(self.medians[col])
        return df

    @annotate("FillMedian_fit", color="green", domain="nvt_python")
    def fit(self, columns: ColumnNames, ddf: dd.DataFrame):
        # TODO: Use `method="tidigest"` when crick supports device
        dask_stats = ddf[columns].quantile(q=0.5, method="dask")
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

    def output_column_names(self, columns: ColumnNames) -> ColumnNames:
        output_cols = columns[:]
        if self.add_binary_cols:
            output_cols.extend([f"{col}_filled" for col in columns])
        return output_cols
