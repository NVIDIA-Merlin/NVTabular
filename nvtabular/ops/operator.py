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

from nvtabular.columns import ColumnSelector
from nvtabular.columns.schema import ColumnSchema, Schema
from nvtabular.dispatch import DataFrameType


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


class Operator:
    """
    Base class for all operator classes.
    """

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
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
        if col_selector and col_selector.names:
            input_selector = col_selector
        else:
            input_selector = ColumnSelector(input_schema.column_names)

        output_selector = self.output_column_names(input_selector)
        output_names = (
            output_selector.names
            if isinstance(output_selector, ColumnSelector)
            else output_selector
        )

        # group (old, new) names, grab old if in new columns
        names_original, names_transformed = [], []
        for original_column in col_selector.names:

            # logic works for all ops except TE
            new_cols = [new_column for new_column in output_names if original_column in new_column]
            if new_cols:
                for column in new_cols:
                    if column not in names_transformed:
                        names_original.append(original_column)
                        names_transformed.append(column)

        # if name is same from input schema means it was a used column,
        # in op logic to mutate against selector columns
        new_names = [
            name
            for name in output_names
            if name not in names_transformed and name not in input_schema.column_schemas
        ]
        new_col_schemas = []

        # grab old column schemas and change the names
        for orig_col_name, new_column_name in zip(names_original, names_transformed):
            orig_col_schema = input_schema.column_schemas[orig_col_name]
            new_col_schemas.append(orig_col_schema.with_name(new_column_name))

        # create column Schemas for new columns
        for new_col_name in new_names:
            if new_col_name not in input_schema.column_schemas:

                # needs to be filled in
                new_col_schemas.append(ColumnSchema(new_col_name))

        # add in all tags properties and update dtypes
        final_new_cols = []
        for column in new_col_schemas:
            column = self._add_tags(column)
            # column = self._add_properties(column)
            column = self._update_dtype(column)
            final_new_cols.append(column)
        return Schema(final_new_cols)

    def _add_tags(self, column_schema):
        return column_schema.with_tags(self.output_tags())

    def _add_properties(self, column_schema):
        # get_properties should return the additional properties
        # for target column
        target_column_properties = self.output_properties()[column_schema.name]
        return column_schema.with_properties(target_column_properties)

    def _update_dtype(self, column_schema):
        return column_schema.with_dtype(self.output_dtype())

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
        import nvtabular

        return nvtabular.ColumnSelector(other) >> self

    @property
    def label(self) -> str:
        return self.__class__.__name__

    @property
    def supports(self) -> Supports:
        """Returns what kind of data representation this operator supports"""
        return Supports.CPU_DATAFRAME | Supports.GPU_DATAFRAME

    def inference_initialize(
        self, col_selector: ColumnSelector, model_config: dict
    ) -> Optional[Operator]:
        """Configures this operator for use in inference. May return a different operator to use
        instead of the one configured for use during training"""
        return None
