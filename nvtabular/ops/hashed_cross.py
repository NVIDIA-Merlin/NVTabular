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

from nvtabular.columns import Schema
from nvtabular.dispatch import DataFrameType, _hash_series, annotate

from ..tags import Tags
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
                val ^= _hash_series(df[column])  # or however we want to do this aggregation

            if isinstance(self.num_buckets, dict):
                val = val % self.num_buckets[cross]
            else:
                val = val % self.num_buckets
            new_df["_X_".join(cross)] = val
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    def output_column_names(self, columns):
        return ColumnSelector(["_X_".join(cross) for cross in _nest_columns(columns)])

    def output_tags(self):
        return [Tags.CATEGORICAL]

    def _get_dtypes(self):
        return numpy.int64

    def compute_output_schema(self, input_schema: Schema, col_selector: ColumnSelector) -> Schema:
        # output_schema = Schema()
        # for column_schema in input_schema.apply(col_selector):
        #     output_schema += Schema([self.transformed_schema(column_schema)])

        # return output_schema
        col_selector = self.output_column_names(col_selector)
        for column_name in col_selector.names:
            if column_name not in input_schema.column_schemas:
                input_schema += Schema([column_name])
        return super().compute_output_schema(input_schema, col_selector)


def _nest_columns(columns):
    # if we have a list of flat column names, lets cross the whole group
    if all(isinstance(col, str) for col in columns):
        return [tuple(columns)]
    else:
        return columns
