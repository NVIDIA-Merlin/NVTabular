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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Text

import numpy

# this needs to be before any modules that import protobuf

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from google.protobuf import json_format, text_format  # noqa
from google.protobuf.any_pb2 import Any  # noqa
from google.protobuf.struct_pb2 import Struct  # noqa
from tensorflow_metadata.proto.v0 import schema_pb2  # noqa

from nvtabular.tags import Tags  # noqa


def register_extra_metadata(column_schema, feature):
    filtered_properties = {k: v for k, v in column_schema.properties.items() if k != "domain"}
    msg_struct = Struct()
    # must pack message into "Any" type
    any_pack = Any()
    any_pack.Pack(json_format.ParseDict(filtered_properties, msg_struct))
    # extra_metadata only takes type "Any" messages
    feature.annotation.extra_metadata.add().CopyFrom(any_pack)
    return feature


def register_list(column_schema, feature):
    if str(column_schema._is_list):
        min_length, max_length = None, None
        if "min_length" in column_schema.properties:
            min_length = column_schema.properties["min_length"]
        if "max_length" in column_schema.properties:
            max_length = column_schema.properties["max_length"]
        if min_length and max_length and min_length == max_length:
            shape = schema_pb2.FixedShape()
            dim = shape.dim.add()
            dim.size = min_length
            feature.shape.CopyFrom(shape)
        elif min_length and max_length and min_length < max_length:
            feature.value_count.CopyFrom(schema_pb2.ValueCount(min=min_length, max=max_length))
        else:
            # if no min max available set dummy value, to signal this is list
            feature.value_count.CopyFrom(schema_pb2.ValueCount(min=0, max=0))
    return feature


def set_protobuf_float(column_schema, feature):
    domain = column_schema.properties.get("domain", {})
    feature.float_domain.CopyFrom(
        schema_pb2.FloatDomain(
            name=column_schema.name,
            min=domain.get("min", None),
            max=domain.get("max", None),
        )
    )
    feature.type = schema_pb2.FeatureType.FLOAT
    return feature


def set_protobuf_int(column_schema, feature):
    domain = column_schema.properties.get("domain", {})
    feature.int_domain.CopyFrom(
        schema_pb2.IntDomain(
            name=column_schema.name,
            min=domain.get("min", None),
            max=domain.get("max", None),
            is_categorical=(
                Tags.CATEGORICAL in column_schema.tags
                or Tags.CATEGORICAL.value in column_schema.tags
            ),
        )
    )
    feature.type = schema_pb2.FeatureType.INT
    return feature


def register_dtype(column_schema, feature):
    #  column_schema is a dict, changes are held
    #  TODO: this double check can be refactored
    if column_schema.dtype:
        if column_schema._is_list:
            feature = proto_dict["list"](column_schema, feature)
        if hasattr(column_schema.dtype, "kind"):
            string_name = numpy.core._dtype._kind_name(column_schema.dtype)
        elif hasattr(column_schema.dtype, "item"):
            string_name = type(column_schema.dtype(1).item()).__name__
        elif isinstance(column_schema.dtype, str):
            string_name = column_schema.dtype
        elif hasattr(column_schema.dtype, "__name__"):
            string_name = column_schema.dtype.__name__
        else:
            raise TypeError(f"unsupported dtype for column schema: {column_schema.dtype}")

        if string_name in proto_dict:
            feature = proto_dict[string_name](column_schema, feature)
    return feature


proto_dict = {
    "list": register_list,
    "float": set_protobuf_float,
    "int": set_protobuf_int,
    "uint": set_protobuf_int,
}


def create_protobuf_feature(column_schema):
    feature = schema_pb2.Feature()
    feature.name = column_schema.name
    feature = register_dtype(column_schema, feature)
    annotation = feature.annotation
    annotation.tag.extend(
        [tag.value if hasattr(tag, "value") else tag for tag in column_schema.tags]
    )
    # can be instantiated with no values
    # if  so, unnecessary to dump
    # import pdb; pdb.set_trace()
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

    @staticmethod
    def read_protobuf(schema_path):
        with open(schema_path, "r") as f:
            schema = schema_pb2.Schema()
            text_format.Parse(f.read(), schema)

        return schema

    @classmethod
    def load_protobuf(cls, schema_path) -> "Schema":
        columns = []
        if isinstance(schema_path, (str, Path)):
            if isinstance(schema_path, str):
                schema_path = Path(schema_path)
            if schema_path.is_dir():
                schema_path = schema_path / "schema.pbtxt"
            schema = cls.read_protobuf(schema_path)

        for feat in schema.feature:
            _is_list = False
            dtype = None
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
            shape_name = feat.WhichOneof("shape_type")
            if shape_name:
                _is_list = True
            field_name = feat.WhichOneof("domain_info")
            if field_name:
                domain_values = getattr(feat, field_name)
                # if zero no values were passed
                if domain_values.max > 0:
                    properties["domain"] = {"min": domain_values.min, "max": domain_values.max}
                if feat.type:
                    if feat.type == 2:
                        dtype = numpy.int
                    elif feat.type == 3:
                        dtype = numpy.float
            columns.append(
                ColumnSchema(
                    feat.name, tags=tags, properties=properties, dtype=dtype, _is_list=_is_list
                )
            )

        return Schema(columns)

    def save_protobuf(self, schema_path):
        schema_path = Path(schema_path)
        if not schema_path.is_dir():
            raise ValueError(f"The path provided is not a valid directory: {schema_path}")

        # traverse list of column schema
        schema = schema_pb2.Schema()
        features = []
        for col_name, col_schema in self.column_schemas.items():
            features.append(create_protobuf_feature(col_schema))
        schema.feature.extend(features)

        with open(schema_path / "schema.pbtxt", "w") as f:
            f.write(text_format.MessageToString(schema))
        return self

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

        return Schema({**self.column_schemas, **other.column_schemas})

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
