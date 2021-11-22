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

from nvtabular.graph import BaseOperator
from nvtabular.graph.node import Node
from nvtabular.graph.ops.selection import SelectionOp
from nvtabular.graph.selector import ColumnSelector
from nvtabular.graph.tags import Tags


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
    group = Node(ColumnSelector(["a"]))

    with pytest.raises(TypeError) as exception_info:
        ColumnSelector(group)

    assert "Node" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        ColumnSelector(["a", "b", group])

    assert "Node" in str(exception_info.value)


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


def test_returned_names_are_unique():
    selector = ColumnSelector(["a", "b", "a"])
    assert selector.names == ["a", "b"]

    selector = ColumnSelector([("a", "b"), ("a", "b")])
    assert selector.grouped_names == [("a", "b")]


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


def test_rshift_operator_onto_selector_creates_selection_node():
    selector = ColumnSelector(["a", "b", "c"])
    operator = BaseOperator()

    output_node = selector >> operator

    assert isinstance(output_node, Node)
    assert isinstance(output_node.op, BaseOperator)
    assert output_node._selector is None
    assert len(output_node.parents) == 1

    assert isinstance(output_node.parents[0], Node)
    assert isinstance(output_node.parents[0].op, SelectionOp)
    assert output_node.parents[0]._selector == selector
    assert len(output_node.parents[0].parents) == 0


def test_construct_column_selector_with_tags():
    target_tags = [Tags.CATEGORICAL, "custom"]
    selector = ColumnSelector(tags=target_tags)
    assert selector.tags == target_tags


def test_returned_tags_are_unique():
    selector = ColumnSelector(tags=["a", "b", "a"])
    assert selector.tags == ["a", "b"]


def test_addition_combines_tags():
    selector1 = ColumnSelector(tags=["a", "b", "c"])
    selector2 = ColumnSelector(tags=["g", "h", "i"])
    combined = selector1 + selector2

    assert combined.tags == ["a", "b", "c", "g", "h", "i"]


def test_addition_combines_names_and_tags():
    selector1 = ColumnSelector(["a", "b", "c"])
    selector2 = ColumnSelector(tags=["g", "h", "i"])
    combined = selector1 + selector2

    assert combined.names == ["a", "b", "c"]
    assert combined.tags == ["g", "h", "i"]


def test_addition_enum_tags():
    selector1 = ColumnSelector(tags=["a", "b", "c"])
    combined = selector1 + Tags.CATEGORICAL

    assert combined.tags == ["a", "b", "c", Tags.CATEGORICAL]

    selector2 = ColumnSelector(["a", "b", "c", ["d", "e", "f"]])
    combined = selector2 + Tags.CATEGORICAL

    assert combined._names == ["a", "b", "c"]
    assert combined.subgroups == [ColumnSelector(["d", "e", "f"])]
    assert combined.tags == [Tags.CATEGORICAL]
