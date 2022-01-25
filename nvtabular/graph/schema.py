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
from typing import Dict, Optional, Text

from nvtabular.graph.schema_io.schema_writer_pbtxt import PbTxt_SchemaWriter
from nvtabular.graph.tags import TagSet


@dataclass(frozen=True)
class ColumnSchema:
    """A schema containing metadata of a dataframe column."""

    name: Text
    tags: Optional[TagSet] = field(default_factory=TagSet)
    properties: Optional[Dict[str, any]] = field(default_factory=dict)
    dtype: Optional[object] = None
    _is_list: bool = False
    _is_ragged: bool = False

    def __post_init__(self):
        tags = TagSet(self.tags)
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
            _is_ragged=self._is_ragged,
        )

    def with_tags(self, tags) -> "ColumnSchema":
        return ColumnSchema(
            self.name,
            tags=self.tags.override(tags),
            properties=self.properties,
            dtype=self.dtype,
            _is_list=self._is_list,
            _is_ragged=self._is_ragged,
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
            _is_ragged=self._is_ragged,
        )

    def with_dtype(self, dtype, is_list=None, is_ragged=None):
        is_list = is_list if is_list is not None else self._is_list

        if is_list:
            is_ragged = is_ragged if is_ragged is not None else self._is_ragged
        else:
            is_ragged = False

        return ColumnSchema(
            self.name,
            tags=self.tags,
            properties=self.properties,
            dtype=dtype,
            _is_list=is_list,
            _is_ragged=is_ragged,
        )

    def __merge__(self, other):
        col_schema = self.with_tags(other.tags)
        col_schema = col_schema.with_properties(other.properties)
        col_schema = col_schema.with_dtype(
            other.dtype, is_list=other._is_list, is_ragged=other._is_ragged
        )
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

        selected_schemas = {
            key: self.column_schemas[key] for key in names if self.column_schemas.get(key, None)
        }
        return Schema(selected_schemas)

    def remove_col(self, col_name):
        if col_name in self.column_names:
            del self.column_schemas[col_name]
        return self

    def without(self, col_names):
        return Schema(
            [
                col_schema
                for col_name, col_schema in self.column_schemas.items()
                if col_name not in col_names
            ]
        )

    @classmethod
    def load(cls, schema_path) -> "Schema":
        return PbTxt_SchemaWriter.load(schema_path)

    def write(self, schema_path):
        return PbTxt_SchemaWriter.write(self, schema_path)

    def get(self, col_name, default=None):
        return self.column_schemas.get(col_name, default)

    def __getitem__(self, column_name):
        if isinstance(column_name, str):
            return self.column_schemas[column_name]
        elif isinstance(column_name, list):
            return Schema([self.column_schemas[col_name] for col_name in column_name])

    def __setitem__(self, column_name, column_schema):
        self.column_schemas[column_name] = column_schema

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

        col_schemas = []

        # must account for same columns in both schemas,
        # use the one with more information for each field
        keys_self_not_other = [
            col_name for col_name in self.column_names if col_name not in other.column_names
        ]

        for key in keys_self_not_other:
            col_schemas.append(self.column_schemas[key])

        for col_name, other_schema in other.column_schemas.items():
            if col_name in self.column_schemas:
                # check which one
                self_schema = self.column_schemas[col_name]
                col_schemas.append(self_schema.__merge__(other_schema))
            else:
                col_schemas.append(other_schema)

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
