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
from nvtabular.graph.schema import ColumnSchema, Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.nvt_dtypes import NVTDtype


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

    def __init__(self):
        # keys are output cols, vals are lists of corresponding input cols
        self._column_mapping = {}

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        breakpoint()
        return selector

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        """Given the schemas coming from upstream sources and a column selector for the
        input columns, returns a set of schemas for the input columns this operator will use
        Parameters
        -----------
        root_schema: Schema
            Base schema of the dataset before running any operators.
        parents_schema: Schema
            The combined schemas of the upstream parents feeding into this operator
        deps_schema: Schema
            The combined schemas of the upstream dependencies feeding into this operator
        col_selector: ColumnSelector
            The column selector to apply to the input schema
        Returns
        -------
        Schema
            The schemas of the columns used by this operator
        """
        return parents_schema + deps_schema

    # TODO: Rework the overrides of this method to work the same (no calls to `transformed_schema`)
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

        self.construct_column_mapping(col_selector)

        output_schema = Schema()
        for output_col_name, input_col_names in self._column_mapping.items():
            col_schema = self.compute_column_schema(output_col_name, input_schema[input_col_names])
            output_schema += Schema([col_schema])

        return output_schema

    def construct_column_mapping(self, col_selector):
        for col_name in col_selector.names:
            self._column_mapping[col_name] = [col_name]

    def compute_column_schema(self, col_name, input_schemas):
        col_schema = ColumnSchema(col_name)

        for method in [self._compute_dtype, self._compute_tags, self._compute_properties]:
            col_schema = method(col_schema, input_schemas)

        return col_schema

    # TODO: Override these `_compute` methods in the operators as needed
    def _compute_dtype(self, col_schema, input_schemas):
        source_col_name = input_schemas.column_names[0]
        return col_schema.with_dtype(input_schemas[source_col_name].dtype)

    def _compute_tags(self, col_schema, input_schemas):
        source_col_name = input_schemas.column_names[0]
        return col_schema.with_tags(input_schemas[source_col_name].tags)

    def _compute_properties(self, col_schema, input_schemas):
        source_col_name = input_schemas.column_names[0]
        return col_schema.with_properties(input_schemas[source_col_name].properties)

    # TODO: Remove this method
    def transformed_schema(self, column_schema):
        column_schema = self._add_tags(column_schema)
        column_schema = self._add_properties(column_schema)
        column_schema = self._update_dtype(column_schema)
        return column_schema

    # TODO: Remove this method
    def _add_tags(self, column_schema):
        return column_schema.with_tags(self.output_tags())

    # TODO: Remove this method
    def _add_properties(self, column_schema):
        # get_properties should return the additional properties
        # for target column
        target_column_properties = self.output_properties().get(column_schema.name, None)
        if target_column_properties:
            return column_schema.with_properties(target_column_properties)
        return column_schema

    # TODO: Remove this method
    def _update_dtype(self, column_schema):
        dtype = self.output_dtype()

        if dtype is None:
            return column_schema

        if isinstance(dtype, NVTDtype):
            return column_schema.with_dtype(dtype)
        elif isinstance(dtype, dict) and column_schema.name in dtype:
            if dtype[column_schema.name] is not None:
                return column_schema.with_dtype(dtype[column_schema.name])
            else:
                return column_schema
        else:
            raise TypeError(
                f"Operator {self.__class__.__name__}.output_dtype returned an invalid type: "
                f"{type(dtype)}"
            )

    # # TODO: Remove this method
    # def output_dtype(self):
    #     """
    #     Retrieves a dictionary of format; column_name: column_dtype. For all
    #     input(with output_names) and created columns
    #     """
    #     # return dict of dtypes of all columns transformed and new columns formed
    #     return None

    # # TODO: Remove this method
    # def output_tags(self):
    #     """
    #     Retrieves
    #     """
    #     # returns a dict of column_name: tags to add to the output columns
    #     return []

    # # TODO: Remove this method
    # def output_properties(self):
    #     # returns dict with column_name: properties to add
    #     return {}

    # TODO: Update instructions for how to define custom
    # operators to reflect constructing the column mapping
    # (They should no longer override this method)
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
        return list(self._column_mapping.keys())

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
