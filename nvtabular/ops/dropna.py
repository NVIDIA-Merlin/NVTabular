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


class Dropna(TransformOperator):
    """
    This operation detects missing values, and returns
    a cudf DataFrame with Null entries dropped from it.
    """

    default_in = ALL
    default_out = ALL

    @annotate("Dropna_op", color="darkgreen", domain="nvt_python")
    def apply_op(
        self,
        gdf: cudf.DataFrame,
        columns_ctx: dict,
        input_cols,
        target_cols=["base"],
        stats_context=None,
    ):
        target_columns = self.get_columns(columns_ctx, input_cols, target_cols)
        new_gdf = gdf.dropna(subset=target_columns or None)
        new_gdf.reset_index(drop=True, inplace=True)
        self.update_columns_ctx(columns_ctx, input_cols, new_gdf.columns, target_columns)
        return new_gdf
