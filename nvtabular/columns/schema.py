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

from tensorflow_metadata.proto.v0 import schema_pb2
from google.protobuf import text_format, json_format
from pathlib import Path
from typing import Dict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.any_pb2 import Any


@dataclass(frozen=True)
class ColumnSchema:
    """A schema containing metadata of a dataframe column."""

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)
    properties: Optional[Dict[str, any]] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.name

    def with_name(self, name) -> "ColumnSchema":
        return ColumnSchema(name, tags=self.tags)

    def with_tags(self, tags) -> "ColumnSchema":
        if not isinstance(tags, list):
            tags = [tags]

        tags = list(set(list(self.tags) + tags))

        return ColumnSchema(self.name, tags=tags, properties=self.properties)
    
    def with_properties(self, properties):
        if not isinstance(properties, dict):
            raise TypeError("properties must be in dict format, key: value")

        # Using new dictionary to avoid passing old ref to new schema
        properties.update(self.properties)

        return ColumnSchema(self.name, tags=self.tags, properties=properties)

    def __eq__(self, other):
        if not isinstance(other, ColumnSchema):
            return False
        if self.name == other.name and self.tags == other.tags and len(other.properties) == len(self.properties):
            #iterate through to ensure ALL keys AND values equal
            for prop_name, prop_value in other.properties.items():
                if prop_name not in self.properties or self.properties[prop_name] != prop_value:
                    return False
            return True
        return False

class ColumnSchemaSet:
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

        return ColumnSchemaSet(selected_schemas)

    def select_by_name(self, names):
        if isinstance(names, str):
            names = [names]

        selected_schemas = {key: self.column_schemas[key] for key in names}
        return ColumnSchemaSet(selected_schemas)

    @staticmethod
    def read_schema_protobuf(schema_path):
        with open(schema_path, "r") as f:
            schema = schema_pb2.Schema()
            text_format.Parse(f.read(), schema)

        return schema

    @classmethod
    def from_schema_protobuf(cls, schema) -> "DatasetSchema":
        if isinstance(schema, (str, Path)):
            schema = cls.read_schema_protobuf(schema)

        columns = []
        for feat in schema.feature:
            tags = feat.annotation.tag
            if feat.HasField("value_count"):
                tags = list(tags) + Tag.LIST.value if tags else Tag.LIST.value
            # only one item should ever be in extra_metadata
            if len(feat.annotation.extra_metadata) > 1:
                raise ValueError(f"extra_metadata for {feat.name} should only have 1 item, has {len(feat.annotation.extra_metadata)}")
            properties = json_format.MessageToDict(feat.annotation.extra_metadata[0])["value"]
            columns.append(ColumnSchema(feat.name, tags=tags, properties=properties))

        return ColumnSchemaSet(columns)

    def to_schema_protobuf(self, schema_path):
        #traverse list of column schema
        schema = schema_pb2.Schema()
        features = []
        for col_name, col_schema in self.column_schemas.items():
            feature = schema_pb2.Feature()
            feature.name = col_name
            annotation = feature.annotation
            annotation.tag.extend(col_schema.tags)
            # must put dictionary in message 
            msg_struct = Struct()
            # msg_struct.update(col_schema.properties)
            # must pack message into "Any" type
            any_pack = Any()
            any_pack.Pack(json_format.ParseDict(col_schema.properties, msg_struct))
            # extra_metadata only takes type "Any" messages
            annotation.extra_metadata.add().CopyFrom(any_pack)
            features.append(feature)
        schema.feature.extend(features)
        with open(schema_path, "w") as f:
            f.write(text_format.MessageToString(schema))
        return self

    def __eq__(self, other):
        if not isinstance(other, ColumnSchemaSet) or len(self.column_schemas) != len(other.column_schemas):
            return False
        for col_name, col_schema in self.column_schemas.items():
            # if not in or if not the same, Fail
            if col_name not in other.column_schemas or other.column_schemas[col_name] != col_schema:
                return False
        return True

    def __add__(self, other):
        if not isinstance(other, ColumnSchemaSet):
            raise TypeError(f"unsupported operand type(s) for +: 'ColumnSchemaSet' and {type(other)}")

        overlap = [name for name in self.column_schemas.keys() if name in other.column_schemas]

        if overlap:
            raise ValueError(f"Overlapping column schemas detected during addition: {overlap}")

        return ColumnSchemaSet({**self.column_schemas, **other.column_schemas})
