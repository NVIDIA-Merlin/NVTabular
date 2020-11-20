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

from .median import Median
from .operator import CONT
from .transform_operator import DFOperator


class FillMissing(DFOperator):
    """
    This operation replaces missing values with a constant pre-defined value

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add FillMissing to the workflow for continuous columns and specify the fill value
        # Default is 0
        proc.add_cont_feature(nvt.ops.FillMissing(fill_val=100))

    Parameters
    -----------
    fill_val : float, default 0
        The constant value to replace missing values with.
    columns : list of str, default None
        Continuous columns to target for this op. If None, the operation will target all known
        continuous columns.
    replace : bool, default True
        Whether to replace existing columns or create new ones.
    """

    default_in = CONT
    default_out = CONT

    def __init__(self, fill_val=0, columns=None, replace=True):
        super().__init__(columns=columns, replace=replace)
        self.fill_val = fill_val

    @property
    def req_stats(self):
        return []

    @annotate("FillMissing_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names:
            return gdf
        z_gdf = gdf[cont_names].fillna(self.fill_val)
        z_gdf.columns = [f"{col}_{self._id}" for col in z_gdf.columns]
        return z_gdf


class FillMedian(DFOperator):
    """
    This operation replaces missing values with the median value for the column.

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add FillMedian to the workflow for continuous columns
        proc.add_cont_feature(nvt.ops.FillMedian())

    Parameters
    -----------
    columns : list of str, default None
        Continuous columns to target for this op. If None, the operation will target all known
        continuous columns.
    replace : bool, default True
        Whether to replace existing columns or create new ones.
    """

    default_in = CONT
    default_out = CONT

    @property
    def req_stats(self):
        return [Median(columns=self.columns)]

    @annotate("FillMedian_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        if not target_columns:
            return gdf

        new_gdf = cudf.DataFrame()
        for col in target_columns:
            stat_val = stats_context["medians"][col]
            new_gdf[col] = gdf[col].fillna(stat_val)
        new_gdf.columns = [f"{col}_{self._id}" for col in new_gdf.columns]
        return new_gdf
