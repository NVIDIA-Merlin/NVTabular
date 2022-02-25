#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import json
import logging

import numpy as np

from nvtabular import ColumnSchema, ColumnSelector
from nvtabular.graph.node import _nodify
from nvtabular.graph.schema import Schema
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator

LOG = logging.getLogger("nvt")


class FilterCandidates(PipelineableInferenceOperator):
    def __init__(self, filter_out, input_col=None):
        self.filter_out = _nodify(filter_out)
        self._input_col = input_col
        self._filter_out_col = filter_out

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        filter_out_col = parameters["filter_out_col"]
        input_col = parameters["input_col"]
        return FilterCandidates(filter_out_col, input_col)

    def dependencies(self):
        return self.filter_out

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        input_schema = super().compute_input_schema(
            root_schema, parents_schema, deps_schema, selector
        )

        if len(parents_schema.column_schemas) > 1:
            raise ValueError(
                "More than one input has been detected for this node,"
                / f"inputs received: {input_schema.column_names}"
            )
        if len(deps_schema.column_schemas) > 1:
            raise ValueError(
                "More than one dependency input has been detected"
                / f"for this node, inputs received: {input_schema.column_names}"
            )

        # 1 for deps and 1 for parents
        if len(input_schema.column_schemas) > 2:
            raise ValueError(
                "More than one input has been detected for this node,"
                / f"inputs received: {input_schema.column_names}"
            )

        self._input_col = parents_schema.column_names[0]
        self._filter_out_col = deps_schema.column_names[0]

        return input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return Schema([ColumnSchema("filtered_ids", dtype=np.int32, _is_list=False)])

    def transform(self, df: InferenceDataFrame):
        candidate_ids = df[self._input_col]
        filter_ids = df[self._filter_out_col]

        filtered_results = np.array([candidate_ids[~np.isin(candidate_ids, filter_ids)]]).T
        return InferenceDataFrame({"filtered_ids": filtered_results})

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}
        self_params = {
            "input_col": self._input_col,
            "filter_out_col": self._filter_out_col,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)
