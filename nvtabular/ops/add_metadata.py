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
        self.tags = tags or []
        self.properties = properties or {}

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df

    def output_tags(self):
        return self.tags

    def output_properties(self):
        return self.properties

    def _add_properties(self, column_schema):
        # get_properties should return the additional properties
        # for target column
        target_column_properties = self.output_properties()
        if target_column_properties:
            return column_schema.with_properties(target_column_properties)
        return column_schema
