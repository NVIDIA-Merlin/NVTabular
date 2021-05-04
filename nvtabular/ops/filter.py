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
from typing import Callable, Union

import cudf
from nvtx import annotate

from .operator import ColumnNames, Operator


class Filter(Operator):
    """
    Filters rows from the dataset. This works by taking a callable that accepts
    a dataframe, and returns a dataframe with unwanted rows filtered out.

    For example to filter out all rows that have a negative value in the ``a`` column::

        filtered = cont_names >> ops.Filter(f=lambda df: df["a"] >=0)
        processor = nvtabular.Workflow(filtered)

    Parameters
    -----------
    f : callable
        Defines a function that takes a dataframe as an argument, and returns a new
        dataframe with unwanted rows filtered out.
    """

    def __init__(self, f: Callable[[cudf.DataFrame], Union[cudf.DataFrame, cudf.Series]]):
        super().__init__()
        if f is None:
            raise ValueError("f cannot be None. Filter op applies f to dataframe")
        self.f = f

    @annotate("Filter_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, gdf: cudf.DataFrame) -> cudf.DataFrame:
        filtered = self.f(gdf)
        if isinstance(filtered, cudf.DataFrame):
            new_gdf = filtered
        elif isinstance(filtered, cudf.Series) and filtered.dtype == bool:
            new_gdf = gdf[filtered]
        else:
            raise ValueError(f"Invalid output from filter op: f{filtered.__class__}")

        new_gdf.reset_index(drop=True, inplace=True)
        return new_gdf

    transform.__doc__ = Operator.transform.__doc__
