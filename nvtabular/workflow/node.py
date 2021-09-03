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

    @property
    def dependency_schema(self):
        schema = Schema()
        for dep in self.dependencies:
            if isinstance(dep, ColumnSelector):
                schema += Schema(dep.names)
            elif isinstance(dep, WorkflowNode):
                schema += dep.output_schema
            elif isinstance(dep, list):
                for nested_dep in dep:
                    if isinstance(nested_dep, ColumnSelector):
                        schema += Schema(nested_dep.names)
                    elif isinstance(nested_dep, WorkflowNode):
                        schema += nested_dep.output_schema
        return schema

    @property
    def parents_schema(self):
        return sum([parent.output_schema for parent in self.parents], Schema())

    def compute_schemas(self, root_schema):
        # If parent is an addition node, we may need to propagate grouping
        # unless we're a node that already has a selector
        if not self.selector:
            if (
                len(self.parents) == 1
                and isinstance(self.parents[0].op, internal.ConcatColumns)
                and self.parents[0].selector
                and self.parents[0].selector.names
            ):

                self.selector = self.parents[0].selector

        # If we have a selector, apply it to upstream schemas from nodes/dataset
        if self.selector:
            upstream_schema = Schema()
            upstream_schema += root_schema
            upstream_schema += self.parents_schema
            upstream_schema += self.dependency_schema
            self.input_schema = upstream_schema.apply(self.selector)
        else:
            # If we don't have a selector but we're an addition node,
            if isinstance(self.op, ConcatColumns):
                upstream_selector = ColumnSelector()

                for parent in self.parents:
                    upstream_selector += ColumnSelector(parent.output_schema.column_names)

                for dep in self.dependencies:
                    if isinstance(dep, WorkflowNode):
                        upstream_selector += ColumnSelector(dep.output_schema.column_names)
                    elif isinstance(dep, ColumnSelector):
                        upstream_selector += dep
                    elif isinstance(dep, list):
                        subgroup_selector = ColumnSelector()
                        for elem in dep:
                            if isinstance(elem, WorkflowNode):
                                subgroup_selector += ColumnSelector(elem.output_schema.column_names)
                            elif isinstance(elem, ColumnSelector):
                                subgroup_selector += elem

                        upstream_selector += ColumnSelector(subgroups=[subgroup_selector])

                if upstream_selector.names:
                    self.selector = upstream_selector

                # For addition nodes, some of the operands are parents and
                # others are dependencies so grab schemas from both
                self.input_schema = self.parents_schema + self.dependency_schema

            # If we're a subtraction node, we have to do some gymnastics to compute
            # the schema, because operands may be in the parents or the dependencies
            # or both
            elif isinstance(self.op, SubsetColumns):
                operands = self.parents + self.dependencies
                left_operand = operands.pop(0)

                if isinstance(left_operand, WorkflowNode):
                    left_operand_schema = left_operand.output_schema
                else:
                    left_operand_schema = Schema(left_operand.names)

                operands_schema = Schema()
                for operand in operands:
                    if isinstance(operand, WorkflowNode):
                        operands_schema += operand.output_schema
                    else:
                        operands_schema += Schema(operand.names)

                self.input_schema = left_operand_schema - operands_schema

            # If none of the above apply, then we don't have a selector
            # and we're not an add or sub node, so our input is just the
            # parents output
            else:
                self.input_schema = self.parents_schema

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
                else:
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
            converted_elem = []
            for elem in other:
                if not isinstance(elem, (ColumnSelector, WorkflowNode)):
                    elem = ColumnSelector(elem)
                converted_elem.append(elem)
            other = converted_elem
        elif not isinstance(other, (ColumnSelector, WorkflowNode)):
            other = ColumnSelector(other)

        if isinstance(other, WorkflowNode) and isinstance(other.op, internal.ConcatColumns):
            child.dependencies += other.parents
            child.dependencies += other.dependencies
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
    def dependencies_schema(self):
        schema = Schema()
        for dep in self.dependencies:
            if isinstance(dep, ColumnSelector):
                schema += Schema(dep.names)
        return schema

    @property
    def parents_schema(self):
        return sum([parent.output_schema for parent in self.parents], Schema())

    @property
    def input_columns(self):
        if not self.input_schema:
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
        if not self.input_schema:
            raise RuntimeError(
                "The output columns aren't computed until the workflow "
                "is fit to a dataset or input schema."
            )

        return ColumnSelector(self.output_schema.column_names)

    @property
    def dependency_columns(self):
        return ColumnSelector(self.dependencies_schema.column_names)

    @property
    def dependency_nodes(self):
        nodes = []
        for dep in self.dependencies:
            if isinstance(dep, WorkflowNode):
                nodes.append(dep)
            elif isinstance(dep, list):
                for nested_dep in dep:
                    if isinstance(nested_dep, WorkflowNode):
                        nodes.append(nested_dep)
        return nodes

    @property
    def label(self):
        if self.op:
            return self.op.label
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
        for parent in node.parents:
            graph.edge(node_ids[parent], nodeid)

    # add a single 'output' node representing the final state
    output_node_id = str(len(allnodes))
    graph.node(output_node_id, f"output cols=[{workflow_node._cols_repr}]")
    graph.edge(node_ids[workflow_node], output_node_id)
    return graph


def _convert_col(col):
    if isinstance(col, (str, tuple)):
        return col
    elif isinstance(col, list):
        return tuple(col)
    else:
        raise ValueError(f"Invalid column value for WorkflowNode: {col}")
