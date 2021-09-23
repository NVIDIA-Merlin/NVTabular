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

import nvtabular
from nvtabular.tags import Tags


class ColumnSelector:
    """A ColumnSelector describes a group of columns to be transformed by Operators in a
    Workflow. Operators can be applied to the selected columns by shifting (>>) operators
    on to the ColumnSelector, which returns a new WorkflowNode with the transformations
    applied. This lets you define a graph of operations that makes up your Workflow.

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

        if isinstance(self._names, nvtabular.WorkflowNode):
            raise TypeError("ColumnSelectors can not contain WorkflowNodes")

        if isinstance(self._names, str):
            self._names = [self._names]

        if isinstance(self.subgroups, ColumnSelector):
            self.subgroups = [self.subgroups]

        plain_names = []
        for name in self._names:
            if isinstance(name, str):
                plain_names.append(name)
            elif isinstance(name, nvtabular.WorkflowNode):
                raise ValueError("ColumnSelectors can not contain WorkflowNodes")
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
        elif isinstance(other, nvtabular.WorkflowNode):
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

    def __rshift__(self, other):
        # Create a virtual input node to shift onto
        virtual_node = nvtabular.WorkflowNode(self)
        virtual_node.op = nvtabular.ops.internal.Identity()
        new_node = virtual_node >> other
        new_node.selector = self

        # Remove virtual input node from the graph
        virtual_node.children = []
        new_node.parents.remove(virtual_node)
        del virtual_node

        # Return the new node as the (real) input node
        return new_node

    def __eq__(self, other):
        if not isinstance(other, ColumnSelector):
            return False
        return other._names == self._names and other.subgroups == self.subgroups
