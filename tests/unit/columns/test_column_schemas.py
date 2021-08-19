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
import pytest

from nvtabular.columns.schema import ColumnSchema, ColumnSchemaSet
from nvtabular.columns.selector import ColumnSelector


def test_column_schema():
    column = ColumnSchema("name", tags=["tag-1"])

    assert column.name == "name"
    assert column.tags[0] == "tag-1"
    assert column.with_name("a").name == "a"
    assert set(column.with_tags("tag-2").tags) == set(["tag-1", "tag-2"])


def test_schema_set_constructor():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["c", "d", "e"])

    expected = {schema1.name: schema1, schema2.name: schema2}

    schema_set_dict = ColumnSchemaSet(expected)
    schema_set_list = ColumnSchemaSet([schema1, schema2])

    assert schema_set_dict.column_schemas == expected
    assert schema_set_list.column_schemas == expected

    with pytest.raises(TypeError) as exception_info:
        ColumnSchemaSet(12345)

    assert "column_schemas" in str(exception_info.value)


def test_schema_set_select_by_tag():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    schema_set = ColumnSchemaSet([schema1, schema2])

    selected_schema1 = schema_set.select_by_tag("a")
    selected_schema2 = schema_set.select_by_tag("d")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_both = schema_set.select_by_tag("c")
    selected_schema_neither = schema_set.select_by_tag("e")
    selected_schema_multi = schema_set.select_by_tag(["b", "c"])

    assert selected_schema_both.column_schemas == {"col1": schema1, "col2": schema2}
    assert selected_schema_neither.column_schemas == {}
    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}


def test_schema_set_select_by_name():
    schema1 = ColumnSchema("col1", tags=["a", "b", "c"])
    schema2 = ColumnSchema("col2", tags=["b", "c", "d"])

    schema_set = ColumnSchemaSet([schema1, schema2])

    selected_schema1 = schema_set.select_by_name("col1")
    selected_schema2 = schema_set.select_by_name("col2")

    assert selected_schema1.column_schemas == {"col1": schema1}
    assert selected_schema2.column_schemas == {"col2": schema2}

    selected_schema_multi = schema_set.select_by_name(["col1", "col2"])

    assert selected_schema_multi.column_schemas == {"col1": schema1, "col2": schema2}

    with pytest.raises(KeyError) as exception_info:
        schema_set.select_by_name("col3")

    assert "col3" in str(exception_info.value)


def test_schema_sets_can_be_added():
    schema_set1 = ColumnSchemaSet([ColumnSchema("col1"), ColumnSchema("col2")])
    schema_set2 = ColumnSchemaSet([ColumnSchema("col3"), ColumnSchema("col4")])

    result = schema_set1 + schema_set2

    expected = ColumnSchemaSet(
        [ColumnSchema("col1"), ColumnSchema("col2"), ColumnSchema("col3"), ColumnSchema("col4")]
    )

    assert result == expected

    with pytest.raises(ValueError) as exception_info:
        schema_set1 + schema_set1  # pylint: disable=pointless-statement

    assert "Overlapping column schemas" in str(exception_info.value)


def test_construct_schema_set_with_column_names():
    schema_set = ColumnSchemaSet(["x", "y", "z"])
    expected = ColumnSchemaSet([ColumnSchema("x"), ColumnSchema("y"), ColumnSchema("z")])

    assert schema_set == expected


def test_schema_set_column_names():
    schema_set = ColumnSchemaSet(["x", "y", "z"])

    assert schema_set.column_names == ["x", "y", "z"]


def test_applying_selector_to_schema_selects_relevant_columns():
    schema = ColumnSchemaSet(["a", "b", "c", "d", "e"])
    selector = ColumnSelector(["a", "b"])
    result = schema.apply(selector)

    assert result == ColumnSchemaSet(["a", "b"])

    selector = None
    result = schema.apply(selector)

    assert result == schema
