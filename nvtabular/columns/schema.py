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

        tags = list(set(list(self.tags) + tags))

        return ColumnSchema(self.name, tags=tags)


class DatasetSchema:
    """A collection of column schemas for a dataset."""

    def __init__(self, column_schemas=None):
        column_schemas = column_schemas or {}

        if isinstance(column_schemas, dict):
            self.column_schemas = column_schemas
        elif isinstance(column_schemas, list):
            self.column_schemas = {}
            for column_schema in column_schemas:
                if isinstance(column_schema, str):
                    column_schema = ColumnSchema(column_schema)
                self.column_schemas[column_schema.name] = column_schema
        else:
            raise TypeError("The `column_schemas` parameter must be a list or dict.")

    @property
    def column_names(self):
        return list(self.column_schemas.keys())

    def apply(self, selector):
        if selector:
            return self.select_by_name(selector.names)
        else:
            return self

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

    def __eq__(self, other):
        if not isinstance(other, DatasetSchema):
            return False

        if len(self.column_schemas) != len(other.column_schemas):
            return False

        return all(column in other.column_schemas for column in self.column_schemas)

    def __add__(self, other):
        if not isinstance(other, DatasetSchema):
            raise TypeError(f"unsupported operand type(s) for +: 'DatasetSchema' and {type(other)}")

        overlap = [name for name in self.column_schemas.keys() if name in other.column_schemas]

        if overlap:
            raise ValueError(f"Overlapping column schemas detected during addition: {overlap}")

        return DatasetSchema({**self.column_schemas, **other.column_schemas})
