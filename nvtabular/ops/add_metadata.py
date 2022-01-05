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
from nvtabular.dispatch import DataFrameType

from .operator import ColumnSelector, Operator


class AddMetadata(Operator):
    """
    This operator will add user defined tags and properties
    to a Schema.
    """

    def __init__(self, tags=None, properties=None):
        super().__init__()
        self.tags = tags or []
        self.properties = properties or {}

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df

    def _compute_tags(self, col_schema, input_schemas):
        source_col_name = input_schemas.column_names[0]
        return col_schema.with_tags(input_schemas[source_col_name].tags + self.output_tags())

    def _compute_properties(self, col_schema, input_schemas):
        source_col_name = input_schemas.column_names[0]
        return col_schema.with_properties({**input_schemas[source_col_name].properties, **self.output_properties()})

    def output_tags(self):
        return self.tags

    def output_properties(self):
        return self.properties
