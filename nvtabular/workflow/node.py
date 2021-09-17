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
import warnings

from nvtabular.columns import ColumnSelector, Schema
from nvtabular.ops import LambdaOp, Operator, internal
from nvtabular.ops.internal.concat_columns import ConcatColumns
from nvtabular.ops.internal.subset_columns import SubsetColumns


class WorkflowNode:
    """A WorkflowNode is a group of columns that you want to apply the same transformations to.
    WorkflowNode's can be transformed by shifting operators on to them, which returns a new
    WorkflowNode with the transformations applied. This lets you define a graph of operations
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
            warnings.warn(
                'The `["a", "b", "c"] >> ops.Operator` syntax for creating a `ColumnGroup` '
                "has been deprecated in NVTabular 21.09 and will be removed in a future version.",
                FutureWarning,
            )
            selector = ColumnSelector(selector)

        if selector and not isinstance(selector, ColumnSelector):
            raise TypeError("The selector argument must be a list or a ColumnSelector")

        self._selector = selector

    @property
    def selector(self):
        return self._selector

    @selector.setter
    def selector(self, sel):
        if isinstance(sel, list):
            sel = ColumnSelector(sel)

        self._selector = sel

    def compute_schemas(self, root_schema):
        # If parent is an addition node, we may need to propagate grouping
        # unless we're a node that already has a selector
        if not self.selector:
            if (
                len(self.parents) == 1
                and isinstance(self.parents[0].op, internal.ConcatColumns)
                and self.parents[0].selector
                and (self.parents[0].selector.names)
            ):

                self.selector = self.parents[0].selector

        # If we have a selector, apply it to upstream schemas from nodes/dataset
        if self.selector:
            upstream_schema = root_schema + _combine_schemas(self.parents_with_dep_nodes)
            self.input_schema = upstream_schema.apply(self.selector)
        else:
            # If we don't have a selector but we're an addition node,
            if isinstance(self.op, ConcatColumns):
                upstream_selector = _combine_selectors(self.parents)
                upstream_selector += _combine_selectors(self.dependencies)

                if upstream_selector.names:
                    self.selector = upstream_selector

                # For addition nodes, some of the operands are parents and
                # others are dependencies so grab schemas from both
                upstream_schema = root_schema + _combine_schemas(self.parents_with_dep_nodes)
                self.input_schema = upstream_schema.apply(self.selector)

            # If we're a subtraction node, we have to do some gymnastics to compute
            # the schema, because operands may be in the parents or the dependencies
            # or both
            elif isinstance(self.op, SubsetColumns):
                operands = self.parents + self.dependencies
                left_operand = operands.pop(0)

                left_operand_schema = _combine_schemas([left_operand])

                operands_schema = _combine_schemas(operands)

                self.input_schema = left_operand_schema - operands_schema

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
        """Transforms this WorkflowNode by applying an Operator

        Parameters
        -----------
        operators: Operator or callable

        Returns
        -------
        WorkflowNode
        """
        if isinstance(operator, type) and issubclass(operator, Operator):
            # handle case where an operator class is passed
            operator = operator()
        elif callable(operator):
            # implicit lambdaop conversion.
            operator = LambdaOp(operator)

        if not isinstance(operator, Operator):
            raise ValueError(f"Expected operator or callable, got {operator.__class__}")

        child = WorkflowNode()
        child.parents = [self]
        self.children.append(child)
        child.op = operator

        dependencies = operator.dependencies()

        if dependencies:
            if not isinstance(dependencies, collections.abc.Sequence):
                dependencies = [dependencies]

            for dependency in dependencies:
                if isinstance(dependency, WorkflowNode):
                    dependency.children.append(child)
                    child.parents.append(dependency)
                elif not isinstance(dependency, ColumnSelector):
                    dependency = ColumnSelector(dependency)
                child.dependencies.append(dependency)

        return child

    def __add__(self, other):
        """Adds columns from this WorkflowNode with another to return a new WorkflowNode

        Parameters
        -----------
        other: WorkflowNode or str or list of str

        Returns
        -------
        WorkflowNode
        """
        if isinstance(self.op, internal.ConcatColumns):
            child = self
        else:
            # Create a child node
            child = WorkflowNode()
            child.op = internal.ConcatColumns(label="+")

            # Add self as a parent
            self.children.append(child)
            child.parents.append(self)

        # The right operand becomes a dependency
        if isinstance(other, list):
            other = _strs_to_selectors(other)
        elif not isinstance(other, (ColumnSelector, WorkflowNode)):
            other = ColumnSelector(other)

        # If the other node is a `+` node, we want to collapse it into this `+` node to
        # avoid creating a cascade of repeated `+`s that we'd need to optimize out by
        # re-combining them later in order to clean up the graph
        if isinstance(other, WorkflowNode) and isinstance(other.op, internal.ConcatColumns):
            child.dependencies += other.parents + other.dependencies
        else:
            child.dependencies.append(other)

        return child

    # handle the "column_name" + WorkflowNode case
    __radd__ = __add__

    def __sub__(self, other):
        """Removes columns from this WorkflowNode with another to return a new WorkflowNode

        Parameters
        -----------
        other: WorkflowNode or str or list of str
            Columns to remove

        Returns
        -------
        WorkflowNode
        """

        if isinstance(self.op, internal.SubsetColumns):
            child = self
        else:
            # Create a child node
            child = WorkflowNode()
            child.op = internal.SubsetColumns(label="-")

            # Add self as a parent
            self.children.append(child)
            child.parents.append(self)

        # The right operand becomes a dependency
        if not isinstance(other, (ColumnSelector, WorkflowNode)):
            other = ColumnSelector(other)

        child.dependencies.append(other)

        return child

    def __rsub__(self, other):
        # Create a child node
        child = WorkflowNode()
        child.op = internal.SubsetColumns(label="-")

        # The left operand becomes a dependency
        if not isinstance(other, (ColumnSelector, WorkflowNode)):
            other = ColumnSelector(other)

        # Add self as a dependency
        child.dependencies.append(other)
        child.dependencies.append(self)

        return child

    def __getitem__(self, columns):
        """Selects certain columns from this WorkflowNode, and returns a new Columngroup with only
        those columns

        Parameters
        -----------
        columns: str or list of str
            Columns to select

        Returns
        -------
        WorkflowNode
        """
        col_selector = ColumnSelector(columns)
        child = WorkflowNode(col_selector)
        child.parents = [self]
        self.children.append(child)
        child.op = internal.SubsetColumns(label=str(list(columns)))
        return child

    def __repr__(self):
        output = " output" if not self.children else ""
        return f"<WorkflowNode {self.label}{output}>"

    @property
    def parents_with_dep_nodes(self):
        return self.parents + self.dependency_nodes

    @property
    def input_columns(self):
        if self.input_schema is None:
            raise RuntimeError(
                "The input columns aren't computed until the workflow "
                "is fit to a dataset or input schema."
            )

        if self.selector:
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
        return _combine_selectors(self.dependency_selectors)

    @property
    def dependency_nodes(self):
        return _filter_by_type(self.dependencies, WorkflowNode)

    @property
    def dependency_selectors(self):
        return _filter_by_type(self.dependencies, ColumnSelector)

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
        yield current
        # TODO: deduplicate nodes?
        for parent in current.parents:
            queue.append(parent)

        for dep in current.dependency_nodes:
            queue.append(dep)


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
        if isinstance(elem, WorkflowNode):
            combined += elem.output_schema
        elif isinstance(elem, ColumnSelector):
            combined += Schema(elem.names)
        elif isinstance(elem, list):
            combined += _combine_schemas(elem)
    return combined


def _combine_selectors(elements):
    combined = ColumnSelector()
    for elem in elements:
        if isinstance(elem, WorkflowNode):
            combined += ColumnSelector(elem.output_schema.column_names)
        elif isinstance(elem, ColumnSelector):
            combined += elem
        elif isinstance(elem, list):
            combined += ColumnSelector(subgroups=_combine_selectors(elem))
    return combined


def _to_selector(value):
    if not isinstance(value, (ColumnSelector, WorkflowNode)):
        return ColumnSelector(value)
    else:
        return value


def _strs_to_selectors(elements):
    return [_to_selector(elem) for elem in elements]


def _to_graphviz(workflow_node):
    """Converts a WorkflowNode to a GraphViz DiGraph object useful for display in notebooks"""
    from graphviz import Digraph

    graph = Digraph()

    # get all the nodes from parents of this columngroup
    # and add edges between each of them
    allnodes = list(set(iter_nodes([workflow_node])))
    node_ids = {v: str(k) for k, v in enumerate(allnodes)}
    for node, nodeid in node_ids.items():
        graph.node(nodeid, node.label)
        for parent in node.parents_with_dep_nodes:
            graph.edge(node_ids[parent], nodeid)

        full_selector = ColumnSelector()

        if node.selector and not node.parents:
            full_selector += node.selector
        full_selector += sum(node.dependency_selectors, full_selector)

        if full_selector.names:
            selector_id = f"{nodeid}_selector"
            graph.node(selector_id, str(full_selector.names))
            graph.edge(selector_id, nodeid)

    # add a single 'output' node representing the final state
    output_node_id = str(len(allnodes))
    output_string = "output cols"
    if workflow_node._cols_repr:
        output_string += f"=[{workflow_node._cols_repr}]"
    graph.node(output_node_id, output_string)
    graph.edge(node_ids[workflow_node], output_node_id)
    return graph


def _convert_col(col):
    if isinstance(col, (str, tuple)):
        return col
    elif isinstance(col, list):
        return tuple(col)
    else:
        raise ValueError(f"Invalid column value for WorkflowNode: {col}")
