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
from cudf._lib.nvtx import annotate

from .operator import ALL
from .transform_operator import TransformOperator


class LambdaOp(TransformOperator):
    """
    Enables to call Methods to cudf.Series

    Parameters
    -----------
    op_name : str
        name of the operator column. It is used as a post_fix for the
        modified column names (if replace=False)
    f : lambda function
        defines the function executed on dataframe level, expectation is lambda col, gdf: ...
        col is the cudf.Series defined by the context
        gdf is the full cudf.DataFrame
    columns :
    preprocessing : bool, default True
        Sets if this is a pre-processing operation or not
    replace : bool, default True
        Replaces the transformed column with the original input
        if set Yes
    """

    default_in = ALL
    default_out = ALL

    def __init__(self, op_name, f, columns=None, preprocessing=True, replace=True):
        super().__init__(columns=columns, preprocessing=preprocessing, replace=replace)
        if op_name is None:
            raise ValueError("op_name cannot be None. It is required for naming the column.")
        if f is None:
            raise ValueError("f cannot be None. LambdaOp op applies f to dataframe")
        self.f = f
        self.op_name = op_name

    @property
    def _id(self):
        return str(self.op_name)

    @annotate("DFLambda_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        new_gdf = cudf.DataFrame()
        for col in target_columns:
            new_gdf[col] = self.f(gdf[col], gdf)
        new_gdf.columns = [f"{col}_{self._id}" for col in new_gdf.columns]
        return new_gdf
