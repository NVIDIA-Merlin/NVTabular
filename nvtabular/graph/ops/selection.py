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

import logging

from nvtabular.dispatch import DataFrameType
from nvtabular.graph.base_operator import BaseOperator
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector

LOG = logging.getLogger("SelectionOp")


class SelectionOp(BaseOperator):
    def __init__(self, selector=None):
        self.selector = selector
        super().__init__()

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        selector = col_selector or self.selector
        return super()._get_columns(df, selector)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        upstream_schema = root_schema + parents_schema + deps_schema
        return upstream_schema.apply(self.selector)

    def compute_output_schema(self, input_schema: Schema, col_selector: ColumnSelector) -> Schema:
        selector = col_selector or self.selector
        return super().compute_output_schema(input_schema, selector)
