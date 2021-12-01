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
from typing import Dict, List, Optional, Text

from nvtabular.graph.schema_io.schema_writer_pbtxt import PbTxt_SchemaWriter
from nvtabular.graph.tags import Tags


@dataclass(frozen=True)
class ColumnSchema:
    """A schema containing metadata of a dataframe column."""

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)
    properties: Optional[Dict[str, any]] = field(default_factory=dict)
    dtype: Optional[object] = None
    _is_list: bool = False

    def __post_init__(self):
        tags = _normalize_tags(self.tags or [])
        object.__setattr__(self, "tags", tags)

    def __str__(self) -> str:
        return self.name

    def with_name(self, name) -> "ColumnSchema":
        return ColumnSchema(
            name,
            tags=self.tags,
            properties=self.properties,
            dtype=self.dtype,
            _is_list=self._is_list,
        )

    def with_tags(self, tags) -> "ColumnSchema":
        if not isinstance(tags, list):
            tags = [tags]

        tags = list(set(list(self.tags) + tags))

        return ColumnSchema(
            self.name,
            tags=tags,
            properties=self.properties,
            dtype=self.dtype,
            _is_list=self._is_list,
        )

    def with_properties(self, properties):
        if not isinstance(properties, dict):
            raise TypeError("properties must be in dict format, key: value")

        # Using new dictionary to avoid passing old ref to new schema
        properties.update(self.properties)

        return ColumnSchema(
            self.name,
            tags=self.tags,
            properties=properties,
            dtype=self.dtype,
            _is_list=self._is_list,
        )

    def with_dtype(self, dtype, is_list=None):
        is_list = is_list or self._is_list
        return ColumnSchema(
            self.name, tags=self.tags, properties=self.properties, dtype=dtype, _is_list=is_list
        )

    def __merge__(self, other):
        col_schema = self.with_tags(other.tags)
        col_schema = col_schema.with_properties(other.properties)
        col_schema = col_schema.with_dtype(other.dtype)
        col_schema = col_schema.with_name(other.name)
        return col_schema


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
        if selector is not None:
            schema = Schema()
            if selector.names:
                schema += self.select_by_name(selector.names)
            if selector.tags:
                schema += self.select_by_tag(selector.tags)
            return schema
        return self

    def apply_inverse(self, selector):
        if selector:
            return self - self.select_by_name(selector.names)
        return self

    def select_by_tag(self, tags):
        if not isinstance(tags, list):
            tags = [tags]

        selected_schemas = {}

        for _, column_schema in self.column_schemas.items():
            if any(x in column_schema.tags for x in tags):
                selected_schemas[column_schema.name] = column_schema

        return Schema(selected_schemas)

    def select_by_name(self, names):
        if isinstance(names, str):
            names = [names]

        selected_schemas = {key: self.column_schemas[key] for key in names}
        return Schema(selected_schemas)

    @classmethod
    def load(cls, schema_path) -> "Schema":
        return PbTxt_SchemaWriter.load(schema_path)

    def write(self, schema_path):
        return PbTxt_SchemaWriter.write(self, schema_path)

    def __iter__(self):
        return iter(self.column_schemas.values())

    def __len__(self):
        return len(self.column_schemas)

    def __repr__(self):
        return str([col_schema.__dict__ for col_schema in self.column_schemas.values()])

    def __eq__(self, other):
        if not isinstance(other, Schema) or len(self.column_schemas) != len(other.column_schemas):
            return False
        return self.column_schemas == other.column_schemas

    def __add__(self, other):
        if other is None:
            return self
        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for +: 'Schema' and {type(other)}")

        # must account for same columns in both schemas,
        # use the one with more information for each field
        keys_other_not_self = [
            schema for schema in other.column_schemas if schema not in self.column_schemas
        ]
        col_schemas = []
        for col_name, col_schema in self.column_schemas.items():
            if col_name in other.column_schemas:
                # check which one
                other_schema = other.column_schemas[col_name]
                col_schemas.append(col_schema.__merge__(other_schema))
            else:
                col_schemas.append(col_schema)
        for key in keys_other_not_self:
            col_schemas.append(other.column_schemas[key])

        return Schema(col_schemas)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if other is None:
            return self

        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for -: 'Schema' and {type(other)}")

        result = Schema({**self.column_schemas})

        for key in other.column_schemas.keys():
            if key in self.column_schemas.keys():
                result.column_schemas.pop(key, None)

        return result


def _normalize_tags(tags):
    return [Tags[tag.upper()] if tag in Tags._value2member_map_ else tag for tag in tags]
