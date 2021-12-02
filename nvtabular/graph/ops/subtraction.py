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
from nvtabular.graph.selector import ColumnSelector


class SubtractionOp(BaseOperator):
    def __init__(self, selector=None):
        self.selector = selector

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        selector = self.selector or col_selector
        return super()._get_columns(df, selector)
