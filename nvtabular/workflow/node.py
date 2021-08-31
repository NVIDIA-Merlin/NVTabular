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
        parent_outputs_schema = sum([parent.output_schema for parent in self.parents], Schema())

        dependencies_schema = Schema()
        for dep in self.dependencies:
            if isinstance(dep, ColumnSelector):
                dependencies_schema += Schema(dep.names)

        if self.selector:
            upstream_schema = Schema()
            upstream_schema += root_schema
            upstream_schema += parent_outputs_schema
            upstream_schema += dependencies_schema

            self.input_schema = upstream_schema.apply(self.selector)
        else:
            self.input_schema = parent_outputs_schema + dependencies_schema

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
            child.dependencies = set()
            child.dependencies = []
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
        left_arg = self
        right_arg = other

        added_nodes = []
        added_selectors = []

        if isinstance(right_arg, WorkflowNode):
            # If an argument is already an addition node, make it
            # the left arg and combine the right arg into it
            if isinstance(right_arg.op, internal.ConcatColumns):
                left_arg = other
                right_arg = self

            added_nodes.append(right_arg)
        elif isinstance(right_arg, ColumnSelector):
            added_selectors.append(right_arg)
        elif isinstance(right_arg, list):
            list_selector = ColumnSelector()
            for element in right_arg:
                if isinstance(element, WorkflowNode):
                    added_nodes.append(element)
                else:
                    list_selector += element
            added_selectors.append(ColumnSelector(subgroups=list_selector))
        else:
            added_selectors.append(ColumnSelector(right_arg))

        if isinstance(left_arg.op, internal.ConcatColumns):
            child = left_arg
        else:
            child = WorkflowNode()
            child.op = internal.ConcatColumns(label="+")

            left_arg.children.append(child)
            child.parents.append(left_arg)

        for node in added_nodes:
            if isinstance(node.op, internal.ConcatColumns):
                child.parents += node.parents
                for parent in node.parents:
                    parent.children.append(child)
                    parent.children.remove(node)
            else:
                child.parents.append(node)
                node.children.append(child)

        for selector in added_selectors:
            child.dependencies.append(selector)

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
        if isinstance(other, WorkflowNode):
            to_remove = set(other.output_columns)
        elif isinstance(other, str):
            to_remove = {other}
        elif isinstance(other, collections.abc.Sequence):
            to_remove = set(other)
        else:
            raise ValueError(f"Expected WorkflowNode, str, or list of str. Got {other.__class__}")
        new_columns = [c for c in self.output_columns if c not in to_remove]
        child = WorkflowNode(new_columns)
        child.parents = [self]
        self.children.append(child)
        child.op = internal.SubsetColumns(label=f"- {list(to_remove)}")
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
    def input_columns(self):
        return ColumnSelector(self.input_schema.column_names)

    @property
    def output_columns(self):
        return ColumnSelector(self.output_schema.column_names)

    @property
    def dependency_columns(self):
        dependency_cols = []

        if not self.dependencies:
            return ColumnSelector(dependency_cols)

        # Dependencies can be either WorkflowNodes or ColumnSelectors
        # WorkflowNodes are already handled as parents, but we still
        # need to account for the columns in raw (non-node) selectors
        for selector in self.dependencies:
            if isinstance(selector, ColumnSelector):
                dependency_cols += selector.names

        return ColumnSelector(dependency_cols)

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
        cols = ", ".join(map(str, self.selector[:3]))
        if len(self.selector) > 3:
            cols += "..."
        return cols

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
