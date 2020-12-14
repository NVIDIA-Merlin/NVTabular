#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

from nvtabular.ops import LambdaOp, Operator


class ColumnGroup:
    """A ColumnGroup is a group of columns that you want to apply the same transformations to.
    ColumnGroup's can be transformed by shifting operators on to them, which returns a new
    ColumnGroup with the transformations applied. This lets you define a graph of operations
    that makes up your workflow

    Parameters
    ----------
    columns: list of str
        The columns to select from the input Dataset
    """

    def __init__(self, columns):
        if isinstance(columns, str):
            columns = [columns]

        self.columns = columns
        self.parents = []
        self.children = []
        self.op = None
        self.kind = None

    def __rshift__(self, operator):
        """Transforms this ColumnGroup by applying an Operator

        Parameters
        -----------
        operators: Operator or callable

        Returns
        -------
        ColumnGroup
        """
        if isinstance(operator, type) and issubclass(operator, Operator):
            # handle case where an operator class is passed
            operator = operator()
        elif callable(operator):
            # implicit lambdaop conversion.
            operator = LambdaOp(operator)

        if not isinstance(operator, Operator):
            raise ValueError(f"Expected operator or callable, got {operator.__class__}")

        child = ColumnGroup(operator.output_column_names(self.columns))
        child.parents = [self]
        self.children.append(child)
        child.op = operator

        dependencies = operator.dependencies()
        if dependencies:
            if not isinstance(dependencies, collections.abc.Sequence):
                dependencies = [dependencies]

            for dependency in dependencies:
                if not isinstance(dependency, ColumnGroup):
                    dependency = ColumnGroup(dependency)
                dependency.children.append(child)
                child.parents.append(dependency)

        return child

    def __add__(self, other):
        """Adds columns from this ColumnGroup with another to return a new ColumnGroup

        Parameters
        -----------
        other: ColumnGroup or str or list of str

        Returns
        -------
        ColumnGroup
        """
        if isinstance(other, str):
            other = ColumnGroup([other])
        elif isinstance(other, collections.abc.Sequence):
            other = ColumnGroup(other)

        # check if there are any columns with the same name in both column groups
        overlap = set(self.columns).intersection(other.columns)
        if overlap:
            raise ValueError(f"duplicate column names found: {overlap}")

        child = ColumnGroup(self.columns + other.columns)
        child.parents = [self, other]
        child.kind = "+"
        self.children.append(child)
        other.children.append(child)
        return child

    # handle the "column_name" + ColumnGroup case
    __radd__ = __add__

    def __sub__(self, other):
        """Adds columns from this ColumnGroup with another to return a new ColumnGroup

        Parameters
        -----------
        other: ColumnGroup or str or list of str

        Returns
        -------
        ColumnGroup
        """
        if isinstance(other, ColumnGroup):
            to_remove = set(other.columns)
        elif isinstance(other, str):
            to_remove = {other}
        elif isinstance(other, collections.abc.Sequence):
            to_remove = set(other)
        else:
            raise ValueError(f"Expected ColumnGroup, str, or list of str. Got {other.__class__}")
        new_columns = [c for c in self.columns if c not in to_remove]
        child = ColumnGroup(new_columns)
        child.parents = [self]
        self.children.append(child)
        child.kind = f"- {list(to_remove)}"
        return child

    def __repr__(self):
        output = " output" if not self.children else ""
        return f"<ColumnGroup {self.label}{output}>"

    @property
    def label(self):
        if self.op:
            return str(self.op.__class__.__name__)
        elif self.kind:
            return self.kind
        elif not self.parents:
            return f"input cols=[{self._cols_repr}]"
        else:
            return "??"

    @property
    def _cols_repr(self):
        cols = ", ".join(self.columns[:3])
        if len(self.columns) > 3:
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


def _to_graphviz(column_group):
    """ converts a columngroup to a GraphViz DiGraph object useful for display in notebooks """
    from graphviz import Digraph

    column_group = _merge_add_nodes(column_group)
    graph = Digraph()

    # get all the nodes from parents of this columngroup
    # and add edges between each of them
    allnodes = list(set(iter_nodes([column_group])))
    node_ids = {v: str(k) for k, v in enumerate(allnodes)}
    for node, nodeid in node_ids.items():
        graph.node(nodeid, node.label)
        for parent in node.parents:
            graph.edge(node_ids[parent], nodeid)

    # add a single 'output' node representing the final state
    output_node_id = str(len(allnodes))
    graph.node(output_node_id, f"output cols=[{column_group._cols_repr}]")
    graph.edge(node_ids[column_group], output_node_id)
    return graph


def _merge_add_nodes(graph):
    """ merges repeat '+' nodes, leading to nicer looking outputs """
    # lets take a copy to avoid mutating the input
    import copy

    graph = copy.deepcopy(graph)

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
