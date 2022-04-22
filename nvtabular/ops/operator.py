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

from typing import Optional

import nvtabular as nvt
from merlin.core.dispatch import DataFrameType
from merlin.dag import BaseOperator, ColumnSelector


class Operator(BaseOperator):
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

    def inference_initialize(
        self, col_selector: ColumnSelector, model_config: dict
    ) -> Optional[Operator]:
        """Configures this operator for use in inference. May return a different operator to use
        instead of the one configured for use during training"""
        return None

    def create_node(self, selector):
        return nvt.workflow.node.WorkflowNode(selector)
