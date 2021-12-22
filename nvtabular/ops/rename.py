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
from nvtabular.graph.schema import Schema

from ..dispatch import DataFrameType
from .operator import ColumnSelector, Operator


class Rename(Operator):
    """This operation renames columns by one of several methods:

        - using a user defined lambda function to transform column names
        - appending a postfix string to every column name
        - renaming a single column to a single fixed string

    Example usage::

        # Rename columns after LogOp
        cont_features = cont_names >> nvt.ops.LogOp() >> Rename(postfix='_log')
        processor = nvt.Workflow(cont_features)

    Parameters
    ----------
    f : callable, optional
        Function that takes a column name and returns a new column name
    postfix : str, optional
        If set each column name in the output will have this string appended to it
    name : str, optional
        If set, a single input column will be renamed to this string
    """

    def __init__(self, f=None, postfix=None, name=None):
        if not f and postfix is None and name is None:
            raise ValueError("must specify name, f, or postfix, for Rename op")

        self.f = f
        self.postfix = postfix
        self.name = name

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        df = df[col_selector.names]
        df.columns = self.output_column_names(col_selector).names
        return df

    transform.__doc__ = Operator.transform.__doc__

    def compute_output_schema(self, input_schema: Schema, col_selector: ColumnSelector) -> Schema:
        if not col_selector:
            col_selector = ColumnSelector(input_schema.column_names)
        if col_selector.tags:
            tags_col_selector = ColumnSelector(tags=col_selector.tags)
            filtered_schema = input_schema.apply(tags_col_selector)
            col_selector += ColumnSelector(filtered_schema.column_names)

            # zero tags because already filtered
            col_selector._tags = []
        output_schema = Schema()
        for column_name in input_schema.column_schemas:
            new_names = self.output_column_names(ColumnSelector(column_name))
            column_schema = input_schema.column_schemas[column_name]
            for new_name in new_names.names:
                new_column_schema = column_schema.with_name(new_name)
                output_schema += Schema([self.transformed_schema(new_column_schema)])
        return output_schema

    def output_column_names(self, col_selector):
        if self.f:
            return ColumnSelector([self.f(col) for col in col_selector.names])
        elif self.postfix:
            return ColumnSelector([col + self.postfix for col in col_selector.names])
        elif self.name:
            if len(col_selector.names) == 1:
                return ColumnSelector([self.name])
            else:
                raise RuntimeError("Single column name provided for renaming multiple columns")
        else:
            raise RuntimeError("The Rename op requires one of f, postfix, or name to be provided")
