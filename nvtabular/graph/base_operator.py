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

from enum import Flag, auto
from typing import Any, List, Optional, Union

import nvtabular as nvt
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector


class Supports(Flag):
    """Indicates what type of data representation this operator supports for transformations"""

    # cudf dataframe
    CPU_DATAFRAME = auto()
    # pandas dataframe
    GPU_DATAFRAME = auto()
    # dict of column name to numpy array
    CPU_DICT_ARRAY = auto()
    # dict of column name to cupy array
    GPU_DICT_ARRAY = auto()


class BaseOperator:
    """
    Base class for all operator classes.
    """

    def compute_output_schema(self, input_schema: Schema, col_selector: ColumnSelector) -> Schema:
        """Given a set of schemas and a column selector for the input columns,
        returns a set of schemas for the transformed columns this operator will produce
        Parameters
        -----------
        input_schema: Schema
            The schemas of the columns to apply this operator to
        col_selector: ColumnSelector
            The column selector to apply to the input schema
        Returns
        -------
        Schema
            The schemas of the columns produced by this operator
        """
        if not col_selector:
            col_selector = ColumnSelector(input_schema.column_names)

        if col_selector.tags:
            tags_col_selector = ColumnSelector(tags=col_selector.tags)
            filtered_schema = input_schema.apply(tags_col_selector)
            col_selector += ColumnSelector(filtered_schema.column_names)

            # zero tags because already filtered
            col_selector._tags = []

        col_selector = self.output_column_names(col_selector)

        for column_name in col_selector.names:
            if column_name not in input_schema.column_schemas:
                input_schema += Schema([column_name])

        output_schema = Schema()
        for column_schema in input_schema.apply(col_selector):
            output_schema += Schema([self.transformed_schema(column_schema)])
        return output_schema

    def transformed_schema(self, column_schema):
        column_schema = self._add_tags(column_schema)
        column_schema = self._add_properties(column_schema)
        column_schema = self._update_dtype(column_schema)
        return column_schema

    def _add_tags(self, column_schema):
        return column_schema.with_tags(self.output_tags())

    def _add_properties(self, column_schema):
        # get_properties should return the additional properties
        # for target column
        target_column_properties = self.output_properties().get(column_schema.name, None)
        if target_column_properties:
            return column_schema.with_properties(target_column_properties)
        return column_schema

    def _update_dtype(self, column_schema):
        if self.output_dtype():
            return column_schema.with_dtype(self.output_dtype())
        return column_schema

    def output_dtype(self):
        """
        Retrieves a dictionary of format; column_name: column_dtype. For all
        input(with output_names) and created columns
        """
        # return dict of dtypes of all columns transformed and new columns formed
        return None

    def output_tags(self):
        """
        Retrieves
        """
        # returns a dict of column_name: tags to add to the output columns
        return []

    def output_properties(self):
        # returns dict with column_name: properties to add
        return {}

    def output_column_names(self, col_selector: ColumnSelector) -> ColumnSelector:
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
        return col_selector

    def dependencies(self) -> Optional[List[Union[str, Any]]]:
        """Defines an optional list of column dependencies for this operator. This lets you consume columns
        that aren't part of the main transformation workflow.

        Returns
        -------
        str, list of str or ColumnSelector, optional
            Extra dependencies of this operator. Defaults to None
        """
        return None

    def __rrshift__(self, other):
        return nvt.ColumnSelector(other) >> self

    @property
    def label(self) -> str:
        return self.__class__.__name__

    def create_node(self, selector):
        return nvt.graph.Node(selector)

    @property
    def supports(self) -> Supports:
        """Returns what kind of data representation this operator supports"""
        return Supports.CPU_DATAFRAME | Supports.GPU_DATAFRAME

    def _get_columns(self, df, selector):
        if isinstance(df, dict):
            return {col_name: df[col_name] for col_name in selector.names}
        else:
            return df[selector.names]
