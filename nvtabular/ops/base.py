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
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from nvtabular.dispatch import DataFrameType

if TYPE_CHECKING:
    # avoid circular references
    from nvtabular import ColumnGroup

ColumnNames = List[Union[str, List[str]]]


class Operator:
    """
    Base class for all operator classes.
    """

    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        """Transform the dataframe by applying this operator to the set of input columns

        Parameters
        -----------
        columns: list of str or list of list of str
            The columns to apply this operator to
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """
        raise NotImplementedError

    def output_column_names(self, columns: ColumnNames) -> ColumnNames:
        """Given a set of columns names returns the names of the transformed columns this
        operator will produce

        Parameters
        -----------
        columns: list of str, or list of list of str
            The columns to apply this operator to

        Returns
        -------
        list of str, or list of list of str
            The names of columns produced by this operator
        """
        return columns

    def dependencies(self) -> Optional[List[Union[str, ColumnGroup]]]:
        """Defines an optional list of column dependencies for this operator. This lets you consume columns
        that aren't part of the main transformation workflow.

        Returns
        -------
        str, list of str or ColumnGroup, optional
            Extra dependencies of this operator. Defaults to None
        """
        return None

    def output_tags(self):
        return []

    def __rrshift__(self, other) -> ColumnGroup:
        import nvtabular

        return nvtabular.ColumnGroup(other) >> self

    @property
    def label(self) -> str:
        return self.__class__.__name__


class OperatorBlock(object):
    def __init__(self, *ops, auto_renaming=False, sequential=True):
        self._ops = list(ops) if ops else []
        self._ops_by_name = {}
        self.sequential = sequential
        self.auto_renaming = auto_renaming

    def add(self, op, name=None):
        self._ops.append(op)
        if name:
            self._ops_by_name[name] = op

        return op

    def extend(self, ops):
        self._ops.extend(ops)

        return ops

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ops[key]

        return self._ops_by_name[key]

    @property
    def ops(self):
        return self._ops

    def __call__(self, col_or_cols, add=False):
        x = col_or_cols
        name_parts = []

        if self.sequential:
            for op in self._ops:
                if isinstance(op, Operator):
                    name_parts.append(op.__class__.__name__)
                    x = x >> op
                else:
                    x = op(x)
        else:
            out = None
            for op in self._ops:
                if out:
                    out += col_or_cols >> op
                else:
                    out = col_or_cols >> op
            x = out

        if self.auto_renaming:
            from nvtabular.ops import Rename

            x = x >> Rename(postfix="/" + "/".join(name_parts))

        if add:
            return col_or_cols + x

        return x

    def __rrshift__(self, other):
        return self.__call__(other)

    def copy(self):
        to_return = OperatorBlock(
            *self._ops, auto_renaming=self.auto_renaming, sequential=self.sequential
        )
        to_return._ops_by_name = self._ops_by_name

        return to_return
