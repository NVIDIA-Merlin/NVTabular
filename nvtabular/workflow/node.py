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

from dask.core import flatten

from nvtabular.columns import ColumnSelector
from nvtabular.ops import LambdaOp, Operator


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

    def __init__(self, selector):
        self.parents = []
        self.children = []
        self.op = None
        self.kind = None
        self.dependencies = None

        if isinstance(selector, list):
            warnings.warn(
                'The `["a", "b", "c"] >> ops.Operator` syntax for creating a `ColumnGroup` '
                "has been deprecated in NVTabular 21.09 and will be removed in a future version.",
                FutureWarning,
            )
            selector = ColumnSelector(selector)

        if not isinstance(selector, ColumnSelector):
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

        col_selector = ColumnSelector(operator.output_column_names(self.selector))

        child = WorkflowNode(col_selector)
        child.parents = [self]
        self.children.append(child)
        child.op = operator

        dependencies = operator.dependencies()
        if dependencies:
            child.dependencies = set()
            if not isinstance(dependencies, collections.abc.Sequence):
                dependencies = [dependencies]

            for dependency in dependencies:
                if isinstance(dependency, WorkflowNode):
                    pass
                elif isinstance(dependency, ColumnSelector):
                    dependency = WorkflowNode(dependency)
                else:
                    dependency = WorkflowNode(ColumnSelector(dependency))
                dependency.children.append(child)
                child.parents.append(dependency)
                child.dependencies.add(dependency)

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
        if isinstance(other, WorkflowNode):
            pass
        elif isinstance(other, ColumnSelector):
            other = WorkflowNode(other)
        elif isinstance(other, list):
            new_selector = ColumnSelector([])
            for element in other:
                if isinstance(element, WorkflowNode):
                    new_selector += ColumnSelector([], subgroups=[element.selector])
                else:
                    new_selector += ColumnSelector(element)
            other = sum(other, WorkflowNode(ColumnSelector([])))
            other.selector = new_selector
        else:
            other = WorkflowNode(ColumnSelector(other))

        # check if there are any columns with the same name in both column groups
        overlap = set(self.selector.grouped_names).intersection(other.selector.grouped_names)

        if overlap:
            raise ValueError(f"duplicate column names found: {overlap}")

        child = WorkflowNode(self.selector + other.selector)
        child.parents = [self, other]
        child.kind = "+"
        self.children.append(child)
        other.children.append(child)
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
            to_remove = set(other.selector)
        elif isinstance(other, str):
            to_remove = {other}
        elif isinstance(other, collections.abc.Sequence):
            to_remove = set(other)
        else:
            raise ValueError(f"Expected WorkflowNode, str, or list of str. Got {other.__class__}")
        new_columns = [c for c in self.selector if c not in to_remove]
        child = WorkflowNode(new_columns)
        child.parents = [self]
        self.children.append(child)
        child.kind = f"- {list(to_remove)}"
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
        child.kind = str(columns)
        return child

    def __repr__(self):
        output = " output" if not self.children else ""
        return f"<WorkflowNode {self.label}{output}>"

    @property
    def flattened_columns(self):
        return list(flatten(self.selector, container=tuple))

    @property
    def input_column_names(self):
        """Returns the names of columns in the main chain"""
        dependencies = self.dependencies or set()

        return [
            col
            for parent in self.parents
            for col in parent.selector.grouped_names
            if parent not in dependencies
        ]

    @property
    def label(self):
        if self.op:
            return self.op.label
        elif self.kind:
            return self.kind
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

    workflow_node = _merge_add_nodes(workflow_node)
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


def _merge_add_nodes(graph):
    """Merges repeat '+' nodes, leading to nicer looking outputs"""
    # lets take a copy to avoid mutating the input
    import copy

    graph = copy.copy(graph)

    queue = [graph]
    while queue:
        current = queue.pop()
        if current.kind == "+":
            changed = True
            while changed:
                changed = False
                parents = []
                for i, parent in enumerate(current.parents):
                    if parent.kind == "+" and len(parent.children) == 1:
                        changed = True
                        # disconnect parent, point all the grandparents at current instead
                        parents.extend(parent.parents)
                        for grandparent in parent.parents:
                            grandparent.children = [
                                current if child == parent else child
                                for child in grandparent.children
                            ]
                    else:
                        parents.append(parent)
                current.parents = parents

        queue.extend(current.parents)

    return graph


def _convert_col(col):
    if isinstance(col, (str, tuple)):
        return col
    elif isinstance(col, list):
        return tuple(col)
    else:
        raise ValueError(f"Invalid column value for WorkflowNode: {col}")
