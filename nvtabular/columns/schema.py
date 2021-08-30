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
from pathlib import Path
from typing import Dict, List, Optional, Text

from google.protobuf import json_format, text_format
from google.protobuf.any_pb2 import Any
from google.protobuf.struct_pb2 import Struct
from tensorflow_metadata.proto.v0 import schema_pb2
import numpy


def register_extra_metadata(column_schema, feature):
    msg_struct = Struct()
    # msg_struct.update(col_schema.properties)
    # must pack message into "Any" type
    any_pack = Any()
    any_pack.Pack(json_format.ParseDict(column_schema.properties, msg_struct))
    # extra_metadata only takes type "Any" messages
    feature.annotation.extra_metadata.add().CopyFrom(any_pack)
    return feature

def register_list(column_schema, feature):
    if str(column_schema.dtype) == "list":
        #  Need to verify this behavior  for both pandas + cudf
        column_schema.dtype = dtype.leaf_type
        if min_length in column_schema.properties:
            min_length = column_schema.properties["min_length"]
        if max_length  in column_schema.properties:
            max_length = column_schema.properties["max_length"]
        if min_length == max_length:
            shape = schema_pb2.FixedShape()
            dim = shape.dim.add()
            dim.size = min_length
            feature.shape.CopyFrom(shape)
        else:
            feature.value_count.CopyFrom(schema_pb2.ValueCount(min=min_length, max=max_length))
    return feature

def set_protobuf_float(column_schema, feature):
    if str(column_schema.dtype) == ["float32", "float64"]:
        # add emebeddings, tuple in properties (min, max)
        col_min, col_max = column_schema.properties.pop("domain")
        feature.float_domain.min = col_min
        feature.float_domain.max = col_max
        feature.type = 3
    return  feature

def set_protobuf_int(column_schema, feature):
    if str(column_schema.dtype) == ["int8", "int32", "int64"]:
        # add emebeddings, tuple in properties (min, max)
        col_min, col_max = column_schema.properties.pop("domain")
        feature.int_domain.min = col_min
        feature.int_domain.max = col_max
        feature.type = 2
    return feature

def register_dtype(column_schema, feature):
    #  column_schema is a dict, changes are held
    #  TODO: this double check can be refactored
    if str(column_schema.dtype) == "list":
        feature = proto_dict[column_schema.dtype]
    feature = proto_dict[column_schema.dtype]
    return  feature
    
proto_dict = {
    "list": register_list,
    "float": set_protobuf_float,
    "int": set_protobuf_int,
    "properties": register_extra_metadata,
}

def create_protobuf_feature(column_schema):
    feature = schema_pb2.Feature()
    feature.name = column_schema.name
    if column_schema.dtype:
        feature = register_dtype(column_schema, feature)
    annotation = feature.annotation
    annotation.tag.extend(column_schema.tags)
    # can be instantiated with no values
    # if  so, unnecessary to dump
    if len(column_schema.properties) > 0:
        feature = register_extra_metadata(column_schema, feature)
    return feature


@dataclass(frozen=True)
class ColumnSchema:
    """A schema containing metadata of a dataframe column."""

    name: Text
    tags: Optional[List[Text]] = field(default_factory=list)
    properties: Optional[Dict[str, any]] = field(default_factory=dict)
    dtype: Optional[object] = None

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
        if (
            self.name == other.name
            and self.tags == other.tags
            and len(other.properties) == len(self.properties)
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

    @staticmethod
    def read_protobuf(schema_path):
        with open(schema_path, "r") as f:
            schema = schema_pb2.Schema()
            text_format.Parse(f.read(), schema)

        return schema

    @classmethod
    def load_protobuf(cls, schema) -> "Schema":
        columns = []
        if isinstance(schema, (str, Path)):
            schema = cls.read_protobuf(schema)

        for feat in schema.feature:
            properties = {}
            tags = list(feat.annotation.tag) or []
            # only one item should ever be in extra_metadata
            if len(feat.annotation.extra_metadata) > 1:
                raise ValueError(
                    f"{feat.name}: extra_metadata should have 1 item, has \
                    {len(feat.annotation.extra_metadata)}"
                )
            if feat.annotation.extra_metadata:
                properties = json_format.MessageToDict(feat.annotation.extra_metadata[0])["value"]
            # what domain
            # load the domain values
            # import pdb; pdb.set_trace()

            field_name = feat.WhichOneof("domain_info")
            if field_name:
                domain_values = getattr(feat, field_name)
                min_max = domain_values.min, domain_values.max
                properties["domain"] = min_max
            columns.append(ColumnSchema(feat.name, tags=tags, properties=properties))

        return Schema(columns)

    def save_protobuf(self, schema_path):
        # traverse list of column schema
        schema = schema_pb2.Schema()
        features = []
        for col_name, col_schema in self.column_schemas.items():
            features.append(create_protobuf_feature(col_schema))
        schema.feature.extend(features)
        with open(schema_path, "w") as f:
            f.write(text_format.MessageToString(schema))
        return self

    def __eq__(self, other):
        if not isinstance(other, Schema) or len(self.column_schemas) != len(
            other.column_schemas
        ):
            return False
        for col_name, col_schema in self.column_schemas.items():
            # if not in or if not the same, Fail
            if col_name not in other.column_schemas or other.column_schemas[col_name] != col_schema:
                return False
        return True

    def __add__(self, other):
        if not isinstance(other, Schema):
            raise TypeError(
                f"unsupported operand type(s) for +: 'Schema' and {type(other)}"
            )
        if other is None:
            return self

        if not isinstance(other, Schema):
            raise TypeError(f"unsupported operand type(s) for +: 'Schema' and {type(other)}")

        return Schema({**self.column_schemas, **other.column_schemas})

    def __radd__(self, other):
        return self.__add__(other)
