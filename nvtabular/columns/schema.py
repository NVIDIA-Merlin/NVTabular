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
from typing import List, Optional, Text, Dict


@dataclass(frozen=True)
class ColumnSchema:
    """A schema containing metadata of a dataframe column."""

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)
    properties: Optional[Dict[str, any]] = field(default_factory=dict)
    dtype: Optional[object] = None
    _is_list: bool = False

    def __str__(self) -> str:
        return self.name
    def with_name(self, name) -> "ColumnSchema":
        return ColumnSchema(name, tags=self.tags, properties=self.properties, dtype=self.dtype, _is_list=self._is_list)
    def with_tags(self, tags) -> "ColumnSchema":
        if not isinstance(tags, list):
            tags = [tags]

        tags = list(set(list(self.tags) + tags))

        return ColumnSchema(self.name, tags=tags, properties=self.properties, dtype=self.dtype, _is_list=self._is_list)

    def with_properties(self, properties):
        if not isinstance(properties, dict):
            raise TypeError("properties must be in dict format, key: value")

        # Using new dictionary to avoid passing old ref to new schema
        properties.update(self.properties)

        return ColumnSchema(self.name, tags=self.tags, properties=properties, dtype=self.dtype, _is_list=self._is_list)

    def with_dtype(self, dtype, is_list=None):
        is_list = is_list or self._is_list
        return ColumnSchema(self.name, tags=self.tags, properties=self.properties, dtype=dtype, _is_list=is_list)

    def __eq__(self, other):
        if not isinstance(other, ColumnSchema):
            return False
        if (
            self.name == other.name
            and self.tags == other.tags
            and len(other.properties) == len(self.properties)
            and self._is_list == other._is_list
        ):
            # iterate through to ensure ALL keys AND values equal
            for prop_name, prop_value in other.properties.items():
                if prop_name not in self.properties or self.properties[prop_name] != prop_value:
                    return False
            return True
        return False


class Schema:
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

        return Schema(selected_schemas)

    def select_by_name(self, names):
        if isinstance(names, str):
            names = [names]

        selected_schemas = {key: self.column_schemas[key] for key in names}
        return Schema(selected_schemas)

    def __eq__(self, other):
        if not isinstance(other, Schema):
            return False

        if len(self.column_schemas) != len(other.column_schemas):
            return False

        return all(column in other.column_schemas for column in self.column_schemas)

    def __add__(self, other):
        if other is None:
            return self

        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for +: 'Schema' and {type(other)}")

        return Schema({**self.column_schemas, **other.column_schemas})

    def __radd__(self, other):
        return self.__add__(other)

    def __len__(self):
        return len(self.column_schemas)
