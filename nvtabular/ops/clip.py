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

from .operator import CONT
from .transform_operator import TransformOperator


class Clip(TransformOperator):
    """
    This operation clips continuous values so that they are within a min/max bound.
    For instance by setting the min value to 0, you can replace all negative values with 0.
    This is helpful in cases where you want to log normalize values::

        # clip all continuous columns to be positive only, and then take the log of the clipped
        # columns
        workflow.add_cont_feature([Clip(min_value=0), LogOp()])

    Parameters
    ----------
    min_value : float, default None
        The mininum value to clip values to: values less than this will be replaced with
        this value. Specifying ``None`` means don't apply a minimum threshold.
    max_value : float, default None
        The maximum value to clip values to: values greater than this will be replaced with
        this value. Specifying ``None`` means don't apply a maximum threshold.
    columns : list of str, default None
        Continuous columns to target for this op. If None, the operation will target all known
        continuous columns.
    replace : bool, default False
        Whether to replace existing columns or create new ones.
    """

    default_in = CONT
    default_out = CONT

    def __init__(self, min_value=None, max_value=None, columns=None, replace=True):
        if min_value is None and max_value is None:
            raise ValueError("Must specify a min or max value to clip to")
        super().__init__(columns=columns, replace=replace)
        self.min_value = min_value
        self.max_value = max_value

    @annotate("Clip_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names:
            return gdf

        z_gdf = gdf[cont_names]
        z_gdf.columns = [f"{col}_{self._id}" for col in z_gdf.columns]
        if self.min_value is not None:
            z_gdf[z_gdf < self.min_value] = self.min_value
        if self.max_value is not None:
            z_gdf[z_gdf > self.max_value] = self.max_value

        return z_gdf
