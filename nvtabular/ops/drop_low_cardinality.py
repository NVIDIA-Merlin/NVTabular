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
from merlin.core.dispatch import DataFrameType
from merlin.schema import Schema, Tags

from .operator import ColumnSelector, Operator


class DropLowCardinality(Operator):
    """
    DropLowCardinality drops low cardinality categorical columns. This requires the
    cardinality of these columns to be known in the schema - for instance by
    first encoding these columns using Categorify.
    """

    def __init__(self, min_cardinality=2):
        super().__init__()
        self.min_cardinality = min_cardinality

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        """
        Selects all non-categorical columns and any categorical columns
        of at least the minimum cardinality from the dataframe.

        Parameters
        ----------
        col_selector : ColumnSelector
            The columns to select.
        df : DataFrameType
            The dataframe to transform

        Returns
        -------
        DataFrameType
            Dataframe with only the selected columns.
        """
        return super()._get_columns(df, col_selector)

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        """
        Checks the cardinality of the input columns and drops any categorical
        columns with cardinality less than the specified minimum.

        Parameters
        ----------
        input_schema : Schema
            The current node's input schema
        selector : ColumnSelector
            The current node's selector
        parents_selector : ColumnSelector
            A selector for the output columns of the current node's parents
        dependencies_selector : ColumnSelector
            A selector for the output columns of the current node's dependencies

        Returns
        -------
        ColumnSelector
            Selector that contains all non-categorical columns and any categorical columns
            of at least the minimum cardinality.
        """
        self._validate_matching_cols(input_schema, selector, self.compute_selector.__name__)

        cols_to_keep = [col for col in input_schema if Tags.CATEGORICAL not in col.tags]

        for col in input_schema:
            if Tags.CATEGORICAL in col.tags:
                domain = col.int_domain
                if not domain or domain.max > self.min_cardinality:
                    cols_to_keep.append(col.name)

        return ColumnSelector(cols_to_keep)
