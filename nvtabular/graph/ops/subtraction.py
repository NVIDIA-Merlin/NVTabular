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
from __future__ import annotations

from nvtabular.dispatch import DataFrameType
from nvtabular.graph.base_operator import BaseOperator
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector


class SubtractionOp(BaseOperator):
    def __init__(self, selector=None):
        self.selector = selector
        super().__init__()

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        return ColumnSelector(input_schema.column_names)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        result = None
        if deps_schema.column_schemas:
            result = parents_schema - deps_schema
        else:
            subtraction_selector = self.selector or selector
            result = parents_schema.apply_inverse(subtraction_selector)
        return result

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        selector = self.selector or col_selector
        return super()._get_columns(df, selector)
