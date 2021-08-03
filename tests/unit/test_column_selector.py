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

from nvtabular.column_group import ColumnGroup
from nvtabular.column_selector import ColumnSelector


def test_constructor_works_with_single_strings_and_lists():
    selector1 = ColumnSelector("a")
    assert selector1._names == ["a"]

    selector2 = ColumnSelector(["a", "b", "c"])
    assert selector2._names == ["a", "b", "c"]


def test_constructor_works_with_single_subgroups_and_lists():
    selector1 = ColumnSelector([], subgroups=ColumnSelector("a"))
    assert isinstance(selector1.subgroups, list)
    assert selector1.subgroups[0] == ColumnSelector("a")

    selector2 = ColumnSelector([], subgroups=ColumnSelector(["a", "b", "c"]))
    assert isinstance(selector2.subgroups, list)
    assert selector2.subgroups[0] == ColumnSelector(["a", "b", "c"])


def test_constructor_too_many_level():

    selector0 = ColumnSelector([], subgroups=ColumnSelector(["a", "b", "c"]))
    selector1 = ColumnSelector(["f", "w", "g"], subgroups=selector0)
    # next should break

    test_pass = False
    try:
        ColumnSelector(["h", "i"], subgroups=selector1)
    except AttributeError:
        test_pass = True
    assert test_pass


def test_constructor_rejects_column_groups():
    group = ColumnGroup(ColumnSelector(["a"]))

    with pytest.raises(TypeError) as exception_info:
        ColumnSelector(group)

    assert "ColumnGroup" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        ColumnSelector(["a", "b", group])

    assert "ColumnGroup" in str(exception_info.value)


def test_constructor_creates_subgroups_from_nesting():
    selector = ColumnSelector(["a", "b", "c", ["d", "e", "f"]])

    assert selector._names == ["a", "b", "c"]
    assert selector.subgroups == [ColumnSelector(["d", "e", "f"])]

    selector = ColumnSelector(["a", "b", "c", ("d", "e", "f")])

    assert selector._names == ["a", "b", "c"]
    assert selector.subgroups == [ColumnSelector(["d", "e", "f"])]


def test_lists_containing_selectors_create_subgroups():
    selector = ColumnSelector(["a", "b", "c", ColumnSelector(["d", "e", "f"])])

    assert len(selector.subgroups) == 1
    assert selector.subgroups == [ColumnSelector(["d", "e", "f"])]
    assert selector.grouped_names == ["a", "b", "c", ("d", "e", "f")]


def test_names_returns_flat_list():
    selector = ColumnSelector(["a", "b", "c"], [ColumnSelector(["d", "e", "f"])])

    assert selector.names == ["a", "b", "c", "d", "e", "f"]


def test_grouped_names_returns_nested_list():
    selector = ColumnSelector(["a", "b", "c"], [ColumnSelector(["d", "e", "f"])])

    assert selector.grouped_names == ["a", "b", "c", ("d", "e", "f")]


def test_addition_combines_names_and_subgroups():
    selector1 = ColumnSelector(["a", "b", "c", ["d", "e", "f"]])
    selector2 = ColumnSelector(["g", "h", "i", ["j", "k", "l"]])
    combined = selector1 + selector2

    assert combined._names == ["a", "b", "c", "g", "h", "i"]
    assert combined.subgroups[0]._names == ["d", "e", "f"]
    assert combined.subgroups[1]._names == ["j", "k", "l"]
    assert len(combined.subgroups) == 2


def test_addition_works_with_strings():
    selector = ColumnSelector(["a", "b", "c", "d", "e", "f"])
    combined = selector + "g"

    assert combined.names == ["a", "b", "c", "d", "e", "f", "g"]


def test_addition_works_with_lists_of_strings():
    selector = ColumnSelector(["a", "b", "c"])
    combined = selector + ["d", "e", "f"]

    assert combined.names == ["a", "b", "c", "d", "e", "f"]
