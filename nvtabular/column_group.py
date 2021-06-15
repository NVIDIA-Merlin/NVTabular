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
from enum import Enum

from dask.core import flatten

from nvtabular.ops import LambdaOp, Operator


class DefaultTags(Enum):
    # Feature types
    CATEGORICAL = ["categorical"]
    CONTINUOUS = ["continuous"]
    LIST = ["list"]
    IMAGE = ["image"]
    TEXT = ["text"]

    # Feature context
    USER = ["user"]
    ITEM = ["item"]
    CONTEXT = ["context"]

    # Target related
    TARGETS = ["target"]
    TARGETS_BINARY = ["target", "binary"]
    TARGETS_REGRESSION = ["target", "regression"]
    TARGETS_MULTI_CLASS = ["target", "multi_class"]


class Tag:
    CATEGORICAL = DefaultTags.CATEGORICAL
    CONTINUOUS = DefaultTags.CONTINUOUS
    LIST = DefaultTags.LIST
    IMAGE = DefaultTags.IMAGE
    TEXT = DefaultTags.TEXT

    # Feature context
    USER = DefaultTags.USER
    ITEM = DefaultTags.ITEM
    CONTEXT = DefaultTags.CONTEXT

    # Target related
    TARGETS = DefaultTags.TARGETS
    TARGETS_BINARY = DefaultTags.TARGETS_BINARY
    TARGETS_REGRESSION = DefaultTags.TARGETS_REGRESSION
    TARGETS_MULTI_CLASS = DefaultTags.TARGETS_MULTI_CLASS

    def __init__(self, *tag):
        self.tags = tag

    @classmethod
    def parse(cls, tag, allow_list=True):
        if allow_list and isinstance(tag, list):
            return Tag(tag)
        elif isinstance(tag, DefaultTags):
            return Tag(tag.value)
        elif isinstance(tag, Tag):
            return tag


class TagAs:
    def __init__(self, tags=None, is_target=False, is_regression_target=False, is_binary_target=False,
                 is_multi_class_target=False):
        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not tags:
            tags = []
        if not isinstance(tags, list):
            tags = [tags]
        if is_target:
            tags.extend(DefaultTags.TARGETS.value)
        if is_regression_target:
            tags.extend(DefaultTags.TARGETS_REGRESSION.value)
        if is_multi_class_target:
            tags.extend(DefaultTags.TARGETS_MULTI_CLASS.value)
        if is_binary_target:
            tags.extend(DefaultTags.TARGETS_BINARY.value)

        self.tags = list(set(tags))


class ColumnGroup:
    """A ColumnGroup is a group of columns that you want to apply the same transformations to.
    ColumnGroup's can be transformed by shifting operators on to them, which returns a new
    ColumnGroup with the transformations applied. This lets you define a graph of operations
    that makes up your workflow

    Parameters
    ----------
    columns: list of (str or tuple of str)
        The columns to select from the input Dataset. The elements of this list are strings
        indicating the column names in most cases, but can also be tuples of strings
        for feature crosses.
    """

    def __init__(self, columns, tags=None):
        self.parents = []
        self.children = []
        self.op = None
        self.kind = None
        self.dependencies = None

        if not tags:
            tags = []
        if isinstance(tags, DefaultTags):
            tags = tags.value
        self.tags = tags

        if isinstance(columns, str):
            columns = [columns]

        # if any of the values we're passed are a columngroup
        # we have to ourselves as a childnode in the graph.
        if any(isinstance(col, ColumnGroup) for col in columns):
            self.columns = []
            self.kind = "[...]"
            for col in columns:
                if not isinstance(col, ColumnGroup):
                    col = ColumnGroup(col)
                else:
                    # we can't handle nesting arbitrarily deep here
                    # only accept non-nested (str) columns here
                    if any(not isinstance(c, str) for c in col.columns):
                        raise ValueError("Can't handle more than 1 level of nested columns")

                col.children.append(self)
                self.parents.append(col)
                self.columns.append(tuple(col.columns))

        else:
            self.columns = [_convert_col(col) for col in columns]

    def __call__(self, operator, **kwargs):
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
        elif isinstance(operator, TagAs):
            self.tags.extend(operator.tags)

            return self

        if not isinstance(operator, Operator):
            raise ValueError(f"Expected operator or callable, got {operator.__class__}")

        child = ColumnGroup(operator.output_column_names(self.columns))
        child.parents = [self]
        self.children.append(child)
        child.op = operator
        child.tags = operator.output_tags()

        dependencies = operator.dependencies()
        if dependencies:
            child.dependencies = set()
            if not isinstance(dependencies, collections.abc.Sequence):
                dependencies = [dependencies]

            for dependency in dependencies:
                if not isinstance(dependency, ColumnGroup):
                    dependency = ColumnGroup(dependency)
                dependency.children.append(child)
                child.parents.append(dependency)
                child.dependencies.add(dependency)

        return child

    def __rshift__(self, operator):
        return self.__call__(operator)

    def to_feature_group(self):
        from nvtabular.feature_group import FeatureGroup

        return FeatureGroup(self.columns)

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
        """Removes columns from this ColumnGroup with another to return a new ColumnGroup

        Parameters
        -----------
        other: ColumnGroup or str or list of str
            Columns to remove

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

    def __getitem__(self, columns):
        """Selects certain columns from this ColumnGroup, and returns a new Columngroup with only
        those columns

        Parameters
        -----------
        columns: str or list of str
            Columns to select

        Returns
        -------
        ColumnGroup
        """
        if isinstance(columns, str):
            columns = [columns]
        if Tag.parse(columns):
            return self.get_tagged(columns)

        child = ColumnGroup(columns)
        child.parents = [self]
        self.children.append(child)
        child.kind = str(columns)
        return child

    def filter_columns(self, filter_fn):
        filtered = [c for c in self.columns if filter_fn(c)]

        return self[filtered]

    def filter_by_namespace(self, namespace):
        return self.filter_columns(lambda c: c.startswith(namespace))

    def get_tagged(self, tags, output_list=False):
        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not isinstance(tags, list):
            tags = [tags]
        output_cols = []
        column_group = _merge_add_nodes(self)

        # get all the nodes from parents of this columngroup
        # and add edges between each of them
        allnodes = list(set(iter_nodes([column_group])))
        for node in allnodes:
            if all(x in node.tags for x in tags):
                output_cols.extend(node.columns)

        if output_list:
            return list(set(output_cols))

        child = ColumnGroup(list(set(output_cols)))
        child.parents = [self]
        self.children.append(child)
        child.kind = f" (tagged={tags})"

        return child

    def embedding_sizes(self):
        queue = [self]
        output = {}
        while queue:
            current = queue.pop()
            if current.op and hasattr(current.op, "get_embedding_sizes"):
                output.update(current.op.get_embedding_sizes(current.columns))
            elif not current.op:

                # only follow parents if its not an operator node (which could
                # transform meaning of the get_embedding_sizes
                queue.extend(current.parents)

        output = {k: v for k, v in output.items() if k in self.columns}

        return output

    def cardinalities(self):
        return {k: v[0] for k, v in self.embedding_sizes().items()}

    def remove_tagged(self, tags):
        to_remove = self.get_tagged(tags)

        return self - to_remove

    def __repr__(self):
        output = " output" if not self.children else ""
        return f"<ColumnGroup {self.label}{output}>"

    @property
    def flattened_columns(self):
        return list(flatten(self.columns, container=tuple))

    @property
    def input_column_names(self):
        """Returns the names of columns in the main chain"""
        dependencies = self.dependencies or set()
        return [
            col for parent in self.parents for col in parent.columns if parent not in dependencies
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
        cols = ", ".join(map(str, self.columns[:3]))
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
    """Converts a ColumnGroup to a GraphViz DiGraph object useful for display in notebooks"""
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
        raise ValueError(f"Invalid column value for ColumnGroup: {col}")
