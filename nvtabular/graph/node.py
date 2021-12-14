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
import collections.abc

from nvtabular.graph.base_operator import BaseOperator
from nvtabular.graph.ops import ConcatColumns, SelectionOp, SubsetColumns, SubtractionOp
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector


class Node:
    """A Node is a group of columns that you want to apply the same transformations to.
    Node's can be transformed by shifting operators on to them, which returns a new
    Node with the transformations applied. This lets you define a graph of operations
    that makes up your workflow

    Parameters
    ----------
    selector: ColumnSelector
        Defines which columns to select from the input Dataset using column names and tags.
    """

    def __init__(self, selector=None):
        self.parents = []
        self.children = []
        self.dependencies = []

        self.op = None
        self.input_schema = None
        self.output_schema = None

        if isinstance(selector, list):
            selector = ColumnSelector(selector)

        if selector and not isinstance(selector, ColumnSelector):
            raise TypeError("The selector argument must be a list or a ColumnSelector")

        if selector:
            self.op = SelectionOp(selector)

        self._selector = selector

    @property
    def selector(self):
        if not self._selector and any(parent._selector for parent in self.parents):
            return _combine_selectors(self.parents)
        else:
            return self._selector

    @selector.setter
    def selector(self, sel):
        if isinstance(sel, list):
            sel = ColumnSelector(sel)

        self._selector = sel

    # These methods must maintain grouping
    def add_dependency(self, dep):
        dep_nodes = _nodify(dep)
        self.dependencies.append(dep_nodes)

    def add_child(self, child):
        child_nodes = _nodify(child)

        if not isinstance(child_nodes, list):
            child_nodes = [child_nodes]

        for child_node in child_nodes:
            child_node.parents.append(self)

        self.children.extend(child_nodes)

    def add_parent(self, parent):
        parent_nodes = _nodify(parent)

        if not isinstance(parent_nodes, list):
            parent_nodes = [parent_nodes]

        for parent_node in parent_nodes:
            parent_node.children.append(self)

        self.parents.extend(parent_nodes)

    def compute_schemas(self, root_schema):
        # If parent is an addition node, we may need to propagate grouping
        # unless we're a node that already has a selector
        if not self.selector:
            if (
                len(self.parents) == 1
                and isinstance(self.parents[0].op, ConcatColumns)
                and self.parents[0].selector
                and (self.parents[0].selector.names)
            ):
                self.selector = self.parents[0].selector

        if isinstance(self.op, ConcatColumns):  # +
            # For addition nodes, some of the operands are parents and
            # others are dependencies so grab schemas from both
            self.selector = _combine_selectors(self.grouped_parents_with_dependencies)
            self.input_schema = _combine_schemas(self.parents_with_dependencies)

        elif isinstance(self.op, SubtractionOp):  # -
            left_operand = _combine_schemas(self.parents)

            if self.dependencies:
                right_operand = _combine_schemas(self.dependencies)
                self.input_schema = left_operand - right_operand
            else:
                self.input_schema = left_operand.apply_inverse(self.op.selector)

            self.selector = ColumnSelector(self.input_schema.column_names)

        elif isinstance(self.op, SubsetColumns):  # []
            left_operand = _combine_schemas(self.parents)
            right_operand = _combine_schemas(self.dependencies)
            self.input_schema = left_operand - right_operand

        # If we have a selector, apply it to upstream schemas from nodes/dataset
        elif isinstance(self.op, SelectionOp):  # ^
            upstream_schema = root_schema + _combine_schemas(self.parents_with_dependencies)
            self.input_schema = upstream_schema.apply(self.selector)

        # If none of the above apply, then we don't have a selector
        # and we're not an add or sub node, so our input is just the
        # parents output
        else:
            self.input_schema = _combine_schemas(self.parents)

        # Then we delegate to the op (if there is one) to compute this node's
        # output schema. If there's no op, then outputs are just the inputs
        if self.op:
            self.output_schema = self.op.compute_output_schema(self.input_schema, self.selector)
        else:
            self.output_schema = self.input_schema

    def __rshift__(self, operator):
        """Transforms this Node by applying an BaseOperator

        Parameters
        -----------
        operators: BaseOperator or callable

        Returns
        -------
        Node
        """
        if isinstance(operator, type) and issubclass(operator, BaseOperator):
            # handle case where an operator class is passed
            operator = operator()

        if not isinstance(operator, BaseOperator):
            raise ValueError(f"Expected operator or callable, got {operator.__class__}")

        child = type(self)()
        child.op = operator
        child.add_parent(self)

        dependencies = operator.dependencies()

        if dependencies:
            if not isinstance(dependencies, collections.abc.Sequence):
                dependencies = [dependencies]

            for dependency in dependencies:
                child.add_dependency(dependency)

        return child

    def __add__(self, other):
        """Adds columns from this Node with another to return a new Node

        Parameters
        -----------
        other: Node or str or list of str

        Returns
        -------
        Node
        """
        if isinstance(self.op, ConcatColumns):
            child = self
        else:
            # Create a child node
            child = type(self)()
            child.op = ConcatColumns(label="+")
            child.add_parent(self)

        # The right operand becomes a dependency
        other_nodes = _nodify(other)
        other_nodes = [other_nodes]

        for other_node in other_nodes:
            # If the other node is a `+` node, we want to collapse it into this `+` node to
            # avoid creating a cascade of repeated `+`s that we'd need to optimize out by
            # re-combining them later in order to clean up the graph
            if not isinstance(other_node, list) and isinstance(other_node.op, ConcatColumns):
                child.dependencies += other_node.grouped_parents_with_dependencies
            else:
                child.add_dependency(other_node)

        return child

    # handle the "column_name" + Node case
    __radd__ = __add__

    def __sub__(self, other):
        """Removes columns from this Node with another to return a new Node

        Parameters
        -----------
        other: Node or str or list of str
            Columns to remove

        Returns
        -------
        Node
        """
        other_nodes = _nodify(other)

        if not isinstance(other_nodes, list):
            other_nodes = [other_nodes]

        child = type(self)()
        child.add_parent(self)
        child.op = SubtractionOp()

        for other_node in other_nodes:
            if isinstance(other_node.op, SelectionOp) and not other_node.parents_with_dependencies:
                child.selector += other_node.selector
                child.op.selector += child.selector
            else:
                child.add_dependency(other_node)

        return child

    def __rsub__(self, other):
        left_operand = _nodify(other)
        right_operand = self

        if not isinstance(left_operand, list):
            left_operand = [left_operand]

        child = type(self)()
        child.add_parent(left_operand)
        child.op = SubtractionOp()

        if (
            isinstance(right_operand.op, SelectionOp)
            and not right_operand.parents_with_dependencies
        ):
            child.selector += right_operand.selector
            child.op.selector += child.selector
        else:
            child.add_dependency(right_operand)

        return child

    def __getitem__(self, columns):
        """Selects certain columns from this Node, and returns a new Columngroup with only
        those columns

        Parameters
        -----------
        columns: str or list of str
            Columns to select

        Returns
        -------
        Node
        """
        col_selector = ColumnSelector(columns)
        child = type(self)(col_selector)
        child.op = SubsetColumns(label=str(list(columns)))
        child.add_parent(self)
        return child

    def __repr__(self):
        output = " output" if not self.children else ""
        return f"<Node {self.label}{output}>"

    @property
    def parents_with_dependencies(self):
        nodes = []
        for node in self.parents + self.dependencies:
            if isinstance(node, list):
                nodes.extend(node)
            else:
                nodes.append(node)

        return nodes

    @property
    def grouped_parents_with_dependencies(self):
        return self.parents + self.dependencies

    @property
    def input_columns(self):
        if self.input_schema is None:
            raise RuntimeError(
                "The input columns aren't computed until the workflow "
                "is fit to a dataset or input schema."
            )

        if (
            self.selector
            and not self.selector.tags
            and all(not selector.tags for selector in self.selector.subgroups)
        ):
            # To maintain column groupings
            return self.selector
        else:
            return ColumnSelector(self.input_schema.column_names)

    @property
    def output_columns(self):
        if self.output_schema is None:
            raise RuntimeError(
                "The output columns aren't computed until the workflow "
                "is fit to a dataset or input schema."
            )

        return ColumnSelector(self.output_schema.column_names)

    @property
    def dependency_schema(self):
        return _combine_schemas(self.dependencies)

    @property
    def dependency_columns(self):
        return _combine_selectors(self.dependencies)

    @property
    def label(self):
        if self.op and hasattr(self.op, "label"):
            return self.op.label
        elif self.op:
            return str(type(self.op))
        elif not self.parents:
            return f"input cols=[{self._cols_repr}]"
        else:
            return "??"

    @property
    def _cols_repr(self):
        if self.input_schema:
            columns = self.input_schema.column_names
        elif self.selector:
            columns = self.selector.names
        else:
            columns = []

        cols_repr = ", ".join(map(str, columns[:3]))
        if len(columns) > 3:
            cols_repr += "..."

        return cols_repr

    @property
    def graph(self):
        return _to_graphviz(self)


def iter_nodes(nodes):
    queue = nodes[:]
    while queue:
        current = queue.pop()
        if isinstance(current, list):
            queue.extend(current)
        else:
            yield current
            # TODO: deduplicate nodes?
            for node in current.parents_with_dependencies:
                queue.append(node)


def _filter_by_type(elements, type_):
    results = []

    for elem in elements:
        if isinstance(elem, type_):
            results.append(elem)
        elif isinstance(elem, list):
            results += _filter_by_type(elem, type_)

    return results


def _combine_schemas(elements):
    combined = Schema()
    for elem in elements:
        if isinstance(elem, Node):
            combined += elem.output_schema
        elif isinstance(elem, ColumnSelector):
            combined += Schema(elem.names)
        elif isinstance(elem, list):
            combined += _combine_schemas(elem)
    return combined


def _combine_selectors(elements):
    combined = ColumnSelector()
    for elem in elements:
        if isinstance(elem, Node):
            if elem.selector:
                selector = elem.op.output_column_names(elem.selector)
            elif elem.output_schema:
                selector = ColumnSelector(elem.output_schema.column_names)
            elif elem.input_schema:
                selector = ColumnSelector(elem.input_schema.column_names)
                selector = elem.op.output_column_names(selector)
            else:
                selector = ColumnSelector()

            combined += selector
        elif isinstance(elem, ColumnSelector):
            combined += elem
        elif isinstance(elem, str):
            combined += ColumnSelector(elem)
        elif isinstance(elem, list):
            combined += ColumnSelector(subgroups=_combine_selectors(elem))
    return combined


def _to_selector(value):
    if not isinstance(value, (ColumnSelector, Node)):
        return ColumnSelector(value)
    else:
        return value


def _strs_to_selectors(elements):
    return [_to_selector(elem) for elem in elements]


def _to_graphviz(output_node):
    """Converts a Node to a GraphViz DiGraph object useful for display in notebooks"""
    from graphviz import Digraph

    graph = Digraph()

    # get all the nodes from parents of this columngroup
    # and add edges between each of them
    allnodes = list(set(iter_nodes([output_node])))
    node_ids = {v: str(k) for k, v in enumerate(allnodes)}
    for node, nodeid in node_ids.items():
        graph.node(nodeid, node.label)
        for parent in node.parents_with_dependencies:
            graph.edge(node_ids[parent], nodeid)

        if node.selector and node.selector.names:
            selector_id = f"{nodeid}_selector"
            graph.node(selector_id, str(node.selector.names))
            graph.edge(selector_id, nodeid)

    # add a single node representing the final state
    final_node_id = str(len(allnodes))
    final_string = "output cols"
    if output_node._cols_repr:
        final_string += f"=[{output_node._cols_repr}]"
    graph.node(final_node_id, final_string)
    graph.edge(node_ids[output_node], final_node_id)
    return graph


def _convert_col(col):
    if isinstance(col, (str, tuple)):
        return col
    elif isinstance(col, list):
        return tuple(col)
    else:
        raise ValueError(f"Invalid column value for Node: {col}")


def _nodify(nodable):
    # TODO: Update to use abstract nodes
    if isinstance(nodable, str):
        return Node(ColumnSelector([nodable]))

    if isinstance(nodable, ColumnSelector):
        return Node(nodable)
    elif isinstance(nodable, Node):
        return nodable
    elif isinstance(nodable, list):
        nodes = [_nodify(node) for node in nodable]
        non_selection_nodes = [node for node in nodes if not node.selector]
        selection_nodes = [node.selector for node in nodes if node.selector]
        selection_nodes = [Node(_combine_selectors(selection_nodes))] if selection_nodes else []
        return non_selection_nodes + selection_nodes
    else:
        raise TypeError(
            "Unsupported type: Cannot convert object " f"of type {type(nodable)} to Node."
        )
