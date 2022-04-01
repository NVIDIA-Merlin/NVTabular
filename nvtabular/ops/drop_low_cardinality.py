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
from merlin.core.dispatch import DataFrameType, annotate
from merlin.schema import Schema, Tags

from .operator import ColumnSelector, Operator


class DropLowCardinality(Operator):
    """
    DropLowCardinality drops low cardinality categorical columns. This requires the
    cardinality of these columns to be known in the schema - for instance by
    first encoding these columns using Categorify.
    """

    def __init__(self, min_cardinality=2):
        super().__init__()
        self.min_cardinality = min_cardinality
        self.to_drop = []

    @annotate("drop_low_cardinality", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df.drop(self.to_drop, axis=1)

    def compute_output_schema(self, input_schema, selector, prev_output_schema=None):
        output_columns = []
        for col in input_schema:
            if Tags.CATEGORICAL in col.tags:
                domain = col.int_domain
                if domain and domain.max <= self.min_cardinality:
                    self.to_drop.append(col.name)
                    continue
            output_columns.append(col)
        return Schema(output_columns)

    transform.__doc__ = Operator.transform.__doc__
    compute_output_schema.__doc__ = Operator.compute_output_schema.__doc__
