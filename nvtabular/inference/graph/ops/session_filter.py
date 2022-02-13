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
from nvtabular.graph.schema import Schema
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator

LOG = logging.getLogger("nvt")


class FilterCandidates(PipelineableInferenceOperator):
    def __init__(self, candidate_col, filter_col):
        self.candidate_col = candidate_col
        self.filter_col = filter_col

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        candidate_col = parameters["candidate_col"]
        filter_col = parameters["filter_col"]
        return FilterCandidates(candidate_col, filter_col)

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return Schema([ColumnSchema("filtered_ids", dtype=np.int32, _is_list=False)])

    def transform(self, df: InferenceDataFrame):
        candidate_ids = df[self.candidate_col]
        filter_ids = df[self.filter_col]

        filtered_results = np.array([candidate_ids[~np.isin(candidate_ids, filter_ids)]]).T
        return InferenceDataFrame({"filtered_ids": filtered_results})

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}
        self_params = {
            "candidate_col": self.candidate_col,
            "filter_col": self.filter_col,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)
