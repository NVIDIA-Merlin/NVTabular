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

from nvtabular.columns import ColumnSelector
from nvtabular.ops import Operator
from nvtabular.workflow import WorkflowNode


def test_constructor_works_with_single_strings_and_lists():
    selector1 = ColumnSelector("a")
    assert selector1._names == ["a"]

    selector2 = ColumnSelector(["a", "b", "c"])
    assert selector2._names == ["a", "b", "c"]


def test_constructor_no_names():
    selector2 = ColumnSelector(["a", "b", "c"])
    selector3 = ColumnSelector(subgroups=selector2)
    assert selector3.subgroups[0] == selector2


def test_constructor_works_with_single_subgroups_and_lists():
    selector1 = ColumnSelector([], subgroups=ColumnSelector("a"))
    assert isinstance(selector1.subgroups, list)
    assert selector1.subgroups[0] == ColumnSelector("a")

    selector2 = ColumnSelector([], subgroups=ColumnSelector(["a", "b", "c"]))
    assert isinstance(selector2.subgroups, list)
    assert selector2.subgroups[0] == ColumnSelector(["a", "b", "c"])


def test_constructor_too_many_level():
    with pytest.raises(AttributeError) as exc_info:
        ColumnSelector(
            ["h", "i"], subgroups=ColumnSelector(names=["b"], subgroups=ColumnSelector(["a"]))
        )
    assert "Too many" in str(exc_info.value)


def test_constructor_rejects_workflow_nodes():
    group = WorkflowNode(ColumnSelector(["a"]))

    with pytest.raises(TypeError) as exception_info:
        ColumnSelector(group)

    assert "WorkflowNode" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        ColumnSelector(["a", "b", group])

    assert "WorkflowNode" in str(exception_info.value)


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


def test_addition_works_with_none():
    selector = ColumnSelector(["a", "b", "c"])
    combined = selector + None

    assert combined.names == ["a", "b", "c"]


def test_rshift_operator_onto_selector_creates_node_with_selector():
    selector = ColumnSelector(["a", "b", "c"])
    operator = Operator()

    output_node = selector >> operator

    assert isinstance(output_node, WorkflowNode)
    assert output_node.selector == selector
    assert output_node.parents == []
