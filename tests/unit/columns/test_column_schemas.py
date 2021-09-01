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
from pathlib import Path

import numpy
import pytest

from nvtabular.columns.schema import ColumnSchema, Schema
from nvtabular.columns.selector import ColumnSelector


@pytest.mark.parametrize("d_types", [numpy.float32, numpy.float64, numpy.uint32, numpy.uint64])
def test_dtype_column_schema(d_types):
    column = ColumnSchema("name", tags=[], properties=[], dtype=d_types)
    assert column.dtype == d_types


def test_column_schema_meta():
    column = ColumnSchema("name", tags=["tag-1"], properties={"p1": "prop-1"})

    assert column.name == "name"
    assert column.tags[0] == "tag-1"
    assert column.with_name("a").name == "a"
    assert set(column.with_tags("tag-2").tags) == set(["tag-1", "tag-2"])
    assert column.with_properties({"p2": "prop-2"}).properties == {"p1": "prop-1", "p2": "prop-2"}
    assert column.with_tags("tag-2").properties == {"p1": "prop-1"}
    assert set(column.with_properties({"p2": "prop-2"}).tags) == set(["tag-1"])

    assert column == ColumnSchema("name", tags=["tag-1"], properties={"p1": "prop-1"})
    # should not be the same no properties
    assert column != ColumnSchema("name", tags=["tag-1"])
    # should not be the same no tags
    assert column != ColumnSchema("name", properties={"p1": "prop-1"})


@pytest.mark.parametrize("props1", [{}, {"p1": "p1", "p2": "p2"}])
@pytest.mark.parametrize("props2", [{}, {"p3": "p3", "p4": "p4"}])
@pytest.mark.parametrize("tags1", [[], ["a", "b", "c"]])
@pytest.mark.parametrize("tags2", [[], ["c", "d", "e"]])
@pytest.mark.parametrize("d_type", [numpy.float32, numpy.float64, numpy.uint32, numpy.uint64])
@pytest.mark.parametrize("list_type", [True, False])
def test_column_schema_set_protobuf(tmpdir, props1, props2, tags1, tags2, d_type, list_type):
    # create a schema
    schema1 = ColumnSchema("col1", tags=tags1, properties=props1, dtype=d_type, _is_list=list_type)
    schema2 = ColumnSchema("col2", tags=tags2, properties=props2, dtype=d_type, _is_list=list_type)
    column_schema_set = Schema([schema1, schema2])
    # write schema out
    schema_path = Path(tmpdir, "test.py")
    column_schema_set = column_schema_set.save_protobuf(schema_path)
    # read schema back in
    target = Schema.load_protobuf(schema_path)
    # compare read to origin
    assert column_schema_set == target


def test_dataset_schema_constructor():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["c", "d", "e"])

    expected = {schema1.name: schema1, schema2.name: schema2}

    ds_schema_dict = Schema(expected)
    ds_schema_list = Schema([schema1, schema2])

    assert ds_schema_dict.column_schemas == expected
    assert ds_schema_list.column_schemas == expected

    with pytest.raises(TypeError) as exception_info:
        Schema(12345)

    assert "column_schemas" in str(exception_info.value)


def test_dataset_schema_select_by_tag():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    ds_schema = Schema([schema1, schema2])

    selected_schema1 = ds_schema.select_by_tag("a")
    selected_schema2 = ds_schema.select_by_tag("d")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_both = ds_schema.select_by_tag("c")
    selected_schema_neither = ds_schema.select_by_tag("e")
    selected_schema_multi = ds_schema.select_by_tag(["b", "c"])

    assert selected_schema_both.column_schemas == {"col1": schema1, "col2": schema2}
    assert selected_schema_neither.column_schemas == {}
    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}


def test_dataset_schema_select_by_name():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    ds_schema = Schema([schema1, schema2])

    selected_schema1 = ds_schema.select_by_name("col1")
    selected_schema2 = ds_schema.select_by_name("col2")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_multi = ds_schema.select_by_name(["col1", "col2"])

    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}

    with pytest.raises(KeyError) as exception_info:
        ds_schema.select_by_name("col3")

    assert "col3" in str(exception_info.value)


def test_dataset_schemas_can_be_added():
    ds1_schema = Schema([ColumnSchema("col1"), ColumnSchema("col2")])
    ds2_schema = Schema([ColumnSchema("col3"), ColumnSchema("col4")])

    result = ds1_schema + ds2_schema

    expected = Schema(
        [ColumnSchema("col1"), ColumnSchema("col2"), ColumnSchema("col3"), ColumnSchema("col4")]
    )

    assert result == expected


def test_schema_can_be_added_to_none():
    schema_set = Schema(["a", "b", "c"])

    assert (schema_set + None) == schema_set
    assert (None + schema_set) == schema_set


def test_construct_schema_with_column_names():
    schema = Schema(["x", "y", "z"])
    expected = Schema([ColumnSchema("x"), ColumnSchema("y"), ColumnSchema("z")])

    assert schema == expected


def test_dataset_schema_column_names():
    ds_schema = Schema(["x", "y", "z"])

    assert ds_schema.column_names == ["x", "y", "z"]


def test_applying_selector_to_schema_selects_relevant_columns():
    schema = Schema(["a", "b", "c", "d", "e"])
    selector = ColumnSelector(["a", "b"])
    result = schema.apply(selector)

    assert result == Schema(["a", "b"])

    selector = None
    result = schema.apply(selector)

    assert result == schema
