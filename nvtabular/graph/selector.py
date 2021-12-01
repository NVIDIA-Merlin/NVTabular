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
from typing import List, Union

import nvtabular as nvt
from nvtabular.graph.tags import Tags


class ColumnSelector:
    """A ColumnSelector describes a group of columns to be transformed by Operators in a
    Graph. Operators can be applied to the selected columns by shifting (>>) operators
    on to the ColumnSelector, which returns a new Node with the transformations
    applied. This lets you define a graph of operations that makes up your Graph.

    Parameters
    ----------
    names: list of (str or tuple of str)
        The columns to select from the input Dataset. The elements of this list are strings
        indicating the column names in most cases, but can also be tuples of strings
        for feature crosses.
    subgroups, optional: list of ColumnSelector objects
        This provides an alternate syntax for grouping column names together (instead
        of nesting tuples inside the list of names)
    """

    def __init__(
        self,
        names: List[str] = None,
        subgroups: List["ColumnSelector"] = None,
        tags: List[Union[Tags, str]] = None,
    ):
        self._names = names if names is not None else []
        self._tags = tags if tags is not None else []
        self.subgroups = subgroups if subgroups is not None else []

        if isinstance(self._names, nvt.graph.Node):
            raise TypeError("ColumnSelectors can not contain Nodes")

        if isinstance(self._names, str):
            self._names = [self._names]

        if isinstance(self.subgroups, ColumnSelector):
            self.subgroups = [self.subgroups]

        plain_names = []
        for name in self._names:
            if isinstance(name, str):
                plain_names.append(name)
            elif isinstance(name, nvt.graph.Node):
                raise ValueError("ColumnSelectors can not contain Nodes")
            elif isinstance(name, ColumnSelector):
                self.subgroups.append(name)
            else:
                self.subgroups.append(ColumnSelector(name))
        self._names = plain_names
        self._nested_check()

    @property
    def tags(self):
        return list(dict.fromkeys(self._tags).keys())

    @property
    def names(self):
        names = []
        names += self._names
        for subgroup in self.subgroups:
            names += subgroup.names

        # Only return unique column names
        return list(dict.fromkeys(names).keys())

    @property
    def grouped_names(self):
        names = []
        names += self._names
        for subgroup in self.subgroups:
            names.append(tuple(subgroup.names))

        # Only return unique grouped column names
        return list(dict.fromkeys(names).keys())

    def _nested_check(self, nests=0):
        if nests > 1:
            raise AttributeError("Too many nested subgroups")
        for col_sel0 in self.subgroups:
            col_sel0._nested_check(nests=nests + 1)

    def __add__(self, other):
        if other is None:
            return self
        elif isinstance(other, nvt.graph.Node):
            return other + self
        elif isinstance(other, ColumnSelector):

            return ColumnSelector(
                self._names + other._names,
                self.subgroups + other.subgroups,
                tags=self._tags + other._tags,
            )
        elif isinstance(other, Tags):
            return ColumnSelector(self._names, self.subgroups, tags=self._tags + [other])
        else:
            if isinstance(other, str):
                other = [other]
            return ColumnSelector(self._names + other, self.subgroups)

    def __radd__(self, other):
        return self + other

    def __rshift__(self, operator):
        if isinstance(operator, type) and issubclass(operator, nvt.graph.BaseOperator):
            # handle case where an operator class is passed
            operator = operator()

        return operator.create_node(self) >> operator

    def __eq__(self, other):
        if not isinstance(other, ColumnSelector):
            return False
        return other._names == self._names and other.subgroups == self.subgroups

    def __bool__(self):
        return bool(self._names or self.subgroups or self.tags)

    def resolve(self, schema):
        """Takes a schema and produces a new selector with selected column names
        how selection occurs (tags, name) does not matter."""
        # get names from tags or names
        root_selector = ColumnSelector(names=self._names, tags=self.tags)
        new_schema = schema.apply(root_selector)
        new_selector = ColumnSelector(new_schema.column_names)
        for group in self.subgroups:
            new_selector.subgroups.append(group.resolve(schema))
        return new_selector
