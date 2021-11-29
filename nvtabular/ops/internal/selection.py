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

from nvtabular.columns import ColumnSelector
from nvtabular.columns.schema import Schema
from nvtabular.dispatch import DataFrameType
from nvtabular.ops import Operator


class SelectionOp(Operator):
    def __init__(self, selector=None):
        self.selector = selector

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        selector = self.selector or col_selector
        return df[selector.names]

    def compute_output_schema(self, input_schema: Schema, col_selector: ColumnSelector) -> Schema:
        selector = self.selector or col_selector
        return super().compute_output_schema(input_schema, selector)

    def output_column_names(self, col_selector: ColumnSelector) -> ColumnSelector:
        selector = self.selector or col_selector
        return super().output_column_names(selector)
