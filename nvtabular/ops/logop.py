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
import numpy as np
from nvtx import annotate

from .operator import CONT
from .transform_operator import TransformOperator


class LogOp(TransformOperator):
    """
    This operator calculates the log of continuous columns. Note that
    to handle the common case of zerofilling null values, this
    calculates ``log(1+x)`` instead of just ``log(x)``.

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add LogOp to the workflow for continuous columns
        proc.add_cont_feature(nvt.ops.LogOp())

    Parameters
    ----------
    columns : list of str, default None
        Continuous columns to target for this op. If None, the operation will target all known
        continuous columns.
    replace : bool, default False
        Whether to replace existing columns or create new ones.
    """

    default_in = CONT
    default_out = CONT

    @annotate("LogOp_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if not cont_names:
            return gdf
        new_gdf = np.log(gdf[cont_names].astype(np.float32) + 1)
        new_cols = [f"{col}_{self._id}" for col in new_gdf.columns]
        new_gdf.columns = new_cols
        return new_gdf
