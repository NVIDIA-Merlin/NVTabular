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

from .operator import ALL
from .transform_operator import TransformOperator


class Filter(TransformOperator):
    """
    Filters rows from the dataset. This works by taking a callable that takes a dataframe,
    and returns a dataframe with unwanted rows filtered out.

    For example to filter out all rows that have a negative value in the ``a`` column::

        op = Filter(lambda df: df["a"] >=0)

    Parameters
    -----------
    f : callable
        Defines a function that takes a dataframe as an argument, and returns a new
        dataframe with unwanted rows filtered out.
    """

    default_in = ALL
    default_out = ALL

    def __init__(self, f):
        super().__init__(replace=True)
        if f is None:
            raise ValueError("f cannot be None. Filter op applies f to dataframe")
        self.f = f

    @annotate("Filter_op", color="darkgreen", domain="nvt_python")
    def apply_op(
        self,
        gdf: cudf.DataFrame,
        columns_ctx: dict,
        input_cols,
        target_cols=["base"],
        stats_context=None,
    ):
        filtered = self.f(gdf)
        if isinstance(filtered, cudf.DataFrame):
            new_gdf = filtered
        elif isinstance(filtered, cudf.Series) and filtered.dtype == bool:
            new_gdf = gdf[filtered]
        else:
            raise ValueError(f"Invalid output from filter op: f{filtered.__class__}")

        new_gdf.reset_index(drop=True, inplace=True)
        return new_gdf
