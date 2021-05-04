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

    def __rrshift__(self, other) -> ColumnGroup:
        import nvtabular

        return nvtabular.ColumnGroup(other) >> self
