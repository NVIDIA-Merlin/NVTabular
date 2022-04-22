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

from merlin.core.dispatch import (
    DataFrameType,
    SeriesType,
    annotate,
    is_dataframe_object,
    is_series_object,
)

from .operator import ColumnSelector, Operator


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

    def __init__(self, f: Callable[[DataFrameType], Union[DataFrameType, SeriesType]]):
        super().__init__()
        if f is None:
            raise ValueError("f cannot be None. Filter op applies f to dataframe")
        self.f = f

    @annotate("Filter_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        filtered = self.f(df)
        if is_dataframe_object(filtered):
            new_df = filtered
        elif is_series_object(filtered) and filtered.dtype == bool:
            new_df = df[filtered]
        else:
            raise ValueError(f"Invalid output from filter op: f{filtered.__class__}")

        new_df.reset_index(drop=True, inplace=True)
        return new_df

    transform.__doc__ = Operator.transform.__doc__
