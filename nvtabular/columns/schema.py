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
from dataclasses import dataclass, field
from typing import List, Optional, Text


@dataclass(frozen=True)
class ColumnSchema:
    """A schema containing metadata of a dataframe column."""

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)

    def __str__(self) -> str:
        return self.name

    def with_name(self, name) -> "ColumnSchema":
        return ColumnSchema(name, tags=self.tags)

    def with_tags(self, tags) -> "ColumnSchema":
        if not isinstance(tags, list):
            tags = [tags]

        tags = list(set(list(self.tags) + list(tags)))

        return ColumnSchema(self.name, tags=tags)


class DatasetSchema:
    """A collection of column schemas for a dataframe."""

    def __init__(self, column_schemas):
        column_schemas = column_schemas or {}

        if isinstance(column_schemas, dict):
            self.column_schemas = column_schemas
        elif isinstance(column_schemas, list):
            self.column_schemas = {}
            for column_schema in column_schemas:
                self.column_schemas[column_schema.name] = column_schema
        else:
            raise TypeError("The `column_schemas` parameter must be a list or dict.")

    def select_by_tag(self, tags):
        if not isinstance(tags, list):
            tags = [tags]

        selected_schemas = {}

        for _, column_schema in self.column_schemas.items():
            if all(x in column_schema.tags for x in tags):
                selected_schemas[column_schema.name] = column_schema

        return DatasetSchema(selected_schemas)

    def select_by_name(self, names):
        if isinstance(names, str):
            names = [names]

        selected_schemas = {key: self.column_schemas[key] for key in names}
        return DatasetSchema(selected_schemas)
