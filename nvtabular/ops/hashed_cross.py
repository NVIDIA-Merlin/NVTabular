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

from typing import Dict, Union

import numpy

from merlin.core.dispatch import DataFrameType, annotate, hash_series

from .operator import ColumnSelector, Operator


class HashedCross(Operator):
    """
    This ops creates hashed cross columns by first combining categorical features
    and hashing the combined feature, then reducing modulo the number of buckets.

    Example usage::

        # Define parameters
        cat_names = [["name-string", "id"]]
        num_buckets = 10

        # Use HashedCross operator to define NVTabular workflow
        hashed_cross = cat_names >> ops.HashedCross(num_buckets)
        processor = nvtabular.Workflow(hashed_cross)

    Parameters
    ----------
    num_buckets : int or dict
        Column-wise modulo to apply after hash function. Note that this
        means that the corresponding value will be the categorical cardinality
        of the transformed categorical feature. That value will be used as the
        number of "hash buckets" for every output feature.
    """

    def __init__(self, num_buckets: Union[int, Dict[str, int]]):
        super().__init__()
        if not isinstance(num_buckets, (int, dict)):
            raise ValueError(f"num_buckets should be an int or dict, found {num_buckets.__class__}")

        self.num_buckets = num_buckets

    @annotate("HashedCross_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        new_df = type(df)()
        for cross in _nest_columns(col_selector.names):
            val = 0
            for column in cross:
                val = hash_series(df[column]) ^ val  # or however we want to do this aggregation
            if isinstance(self.num_buckets, dict):
                val = val % self.num_buckets[cross]
            else:
                val = val % self.num_buckets
            new_df["_X_".join(cross)] = val.astype(self.output_dtype)
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    def column_mapping(self, col_selector):
        column_mapping = {}
        for cross in _nest_columns(col_selector):
            output_col = "_X_".join(cross)
            column_mapping[output_col] = [*cross]

        return column_mapping

    @property
    def output_dtype(self):
        return numpy.int32


def _nest_columns(columns):
    # if we have a list of flat column names, lets cross the whole group
    if isinstance(columns, ColumnSelector):
        columns = columns.names
    if all(isinstance(col, str) for col in columns):
        return [tuple(columns)]
    else:
        return columns
