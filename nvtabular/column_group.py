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
from operator import attrgetter
from typing import Dict, List, Optional, Text, Union

import joblib
from dask.core import flatten

from nvtabular.column import Column, Columns
from nvtabular.ops import LambdaOp, Operator
from nvtabular.tag import DefaultTags, Tag


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

    def __init__(
        self,
        columns: Union[Text, List[Text], Column, List[Column], "ColumnGroup"],
        tags: Optional[Union[List[Text], DefaultTags]] = None,
        properties: Optional[Dict[Text, Text]] = None,
    ):
        self.parents = []
        self.children = []
        self.op = None
        self.kind = None
        self.dependencies = None

        if isinstance(columns, str):
            columns = [columns]

        if not tags:
            tags = []
        if isinstance(tags, DefaultTags):
            tags = tags.value

        self.tags = tags
        self.properties = properties

        # if any of the values we're passed are a columngroup
        # we have to ourselves as a childnode in the graph.
        if any(isinstance(col, ColumnGroup) for col in columns):
            self.columns: Columns = Columns()
            self.kind = "[...]"
            for col in columns:
                if not isinstance(col, ColumnGroup):
                    col = ColumnGroup(col)
                else:
                    # we can't handle nesting arbitrarily deep here
                    # only accept non-nested (str) columns here
                    if any(not isinstance(c, Column) for c in col.columns):
                        raise ValueError("Can't handle more than 1 level of nested columns")

                col.children.append(self)
                self.parents.append(col)
                self.columns.append(tuple(col.columns))

        else:
            columns = [_convert_col(col, tags=tags, properties=properties) for col in columns]
            self.columns: Columns = Columns(columns)

    @property
    def column_names(self):
        return self.columns.names()

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

        if not isinstance(operator, Operator):
            raise ValueError(f"Expected operator or callable, got {operator.__class__}")
        child_columns = operator.output_columns(self.columns)

        child = ColumnGroup(child_columns)
        child.parents = [self]
        self.children.append(child)
        child.op = operator

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
        overlap = self.column_names.intersection(other.column_names)

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
            to_remove = set(other.column_names)
        elif isinstance(other, str):
            to_remove = {other}
        elif isinstance(other, collections.abc.Sequence):
            to_remove = set(other)
        else:
            raise ValueError(f"Expected ColumnGroup, str, or list of str. Got {other.__class__}")
        new_columns = [c for c in self.columns if c.name not in to_remove]
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
        if isinstance(columns, int):
            return self.column_names[columns]

        filtered_columns = [col for col in _convert_col(columns) if col.name in self.column_names]
        child = ColumnGroup(filtered_columns)
        child.parents = [self]
        self.children.append(child)
        child.kind = str(columns)
        return child

    def filter_columns(self, filter_fn, by_name=True):
        if by_name:
            filtered = [c for c in self.columns if filter_fn(c.name)]
        else:
            filtered = [c for c in self.columns if filter_fn(c)]

        return self[filtered]

    def filter_by_namespace(self, namespace):
        return self.filter_columns(lambda c: c.startswith(namespace))

    def get_tagged(self, tags, output_list=False, tags_to_filter=None):
        column_names_to_filter = (
            self.get_tagged(tags_to_filter, output_list=True) if tags_to_filter else []
        )

        if isinstance(tags, DefaultTags):
            tags = tags.value
        if not isinstance(tags, list):
            tags = [tags]
        output_cols = []

        for column in self.flattened_columns:
            if all(x in column.tags for x in tags):
                output_cols.append(column)

        columns = [col for col in output_cols if col.name not in column_names_to_filter]

        if output_list:
            return [col.name for col in columns]

        child = ColumnGroup(columns, tags=tags)
        child.parents = [self]
        self.children.append(child)
        # child.kind = f"tagged={child._tags_repr} " + self._cols_repr

        return child

    def tags_by_column(self):
        outputs = {}

        for col in self.flattened_columns:
            outputs[col.name] = col.tags

        return outputs

    def targets_columns(self):
        return self.get_tagged(Tag.TARGETS, output_list=True)

    def targets_column_group(self):
        return self.get_tagged(Tag.TARGETS, output_list=False)

    def binary_targets_columns(self):
        return self.get_tagged(Tag.TARGETS_BINARY, output_list=True)

    def binary_targets_column_group(self):
        return self.get_tagged(Tag.TARGETS_BINARY, output_list=False)

    def regression_targets_columns(self):
        return self.get_tagged(Tag.TARGETS_REGRESSION, output_list=True)

    def regression_targets_column_group(self):
        return self.get_tagged(Tag.TARGETS_REGRESSION, output_list=False)

    def continuous_columns(self):
        return self.get_tagged(Tag.CONTINUOUS, output_list=True)

    def continuous_column_group(self):
        return self.get_tagged(Tag.CONTINUOUS, output_list=False)

    def categorical_columns(self):
        return self.get_tagged(Tag.CATEGORICAL, output_list=True)

    def categorical_column_group(self):
        return self.get_tagged(Tag.CATEGORICAL, output_list=False)

    def text_columns(self):
        return self.get_tagged(Tag.TEXT, output_list=True)

    def text_column_group(self):
        return self.get_tagged(Tag.TEXT, output_list=False)

    def text_tokenized_columns(self):
        return self.get_tagged(Tag.TEXT_TOKENIZED, output_list=True)

    def text_tokenized_column_group(self):
        return self.get_tagged(Tag.TEXT_TOKENIZED, output_list=False)

    def embedding_sizes(self):
        return self._embedding_sizes_from_op()

    def cardinalities(self):
        return {k: v[0] for k, v in self._embedding_sizes_from_op().items()}

    def _embedding_sizes_from_op(self):
        queue = [self]
        output = {}
        while queue:
            current = queue.pop()
            if current.op and hasattr(current.op, "get_embedding_sizes"):
                output.update(current.op.get_embedding_sizes(current.column_names))
            elif not current.op:

                # only follow parents if its not an operator node (which could
                # transform meaning of the get_embedding_sizes
                queue.extend(current.parents)

        output = {k: v for k, v in output.items() if k in self.columns}

        return output

    def remove_tagged(self, tags):
        to_remove = self.get_tagged(tags)

        return self - to_remove

    def __repr__(self):
        output = " output" if not self.children else ""
        return f"<ColumnGroup {self.label}{output}>"

    @property
    def flattened_column_names(self):
        return list(flatten(self.column_names))

    @property
    def flattened_columns(self):
        return self.columns.flatten()

    @property
    def input_columns(self) -> Columns:
        dependencies = self.dependencies or set()
        return Columns(
            [col for parent in self.parents for col in parent.columns if parent not in dependencies]
        )

    @property
    def input_column_names(self):
        """Returns the names of columns in the main chain"""
        dependencies = self.dependencies or set()
        return [
            tuple(col) if isinstance(col, Columns) else col
            for parent in self.parents
            for col in parent.column_names
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
    def is_transformation(self):
        return bool(self.parents)

    @property
    def id(self):
        return joblib.hash(sorted([n.label for n in self.nodes]))

    @property
    def _cols_repr(self):
        cols = ", ".join(map(str, self.column_names[:3]))
        if len(self.column_names) > 3:
            cols += "..."
        return cols

    @property
    def _tags_repr(self):
        cols = ", ".join(map(str, self.tags[:3]))
        if len(self.tags) > 3:
            cols += "..."
        return cols

    @property
    def graph(self):
        return _to_graphviz(self)

    @property
    def nodes(self):
        return sorted(list(set(iter_nodes([self]))), key=attrgetter("is_transformation", "label"))


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


def _convert_col(col, tags=None, properties=None):
    if not properties:
        properties = {}
    if isinstance(col, Column):
        return col.with_tags(tags).with_properties(**properties)
    elif isinstance(col, str):
        return Column(col, tags=tags, properties=properties)
    elif isinstance(col, (tuple, list)):
        return Columns(tuple(_convert_col(c, tags=tags, properties=properties) for c in col))
    else:
        raise ValueError(f"Invalid column value for ColumnGroup: {col} (type: {type(col)})")
