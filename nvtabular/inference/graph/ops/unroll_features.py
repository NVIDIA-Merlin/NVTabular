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

import cupy as cp
import numpy as np

from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator

LOG = logging.getLogger("nvtabular")


class UnrollFeatures(PipelineableInferenceOperator):
    def __init__(self, candidate_col, unroll_cols, unrolled_prefix=""):
        self.candidate_col = candidate_col
        self.unroll_cols = unroll_cols
        self.unrolled_prefix = unrolled_prefix

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        candidate_col = parameters["candidate_col"]
        unroll_cols = parameters["unroll_cols"]
        unrolled_prefix = parameters["unrolled_prefix"]
        return UnrollFeatures(candidate_col, unroll_cols, unrolled_prefix)

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}
        self_params = {
            "candidate_col": self.candidate_col,
            "unroll_cols": self.unroll_cols,
            "unrolled_prefix": self.unrolled_prefix,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        schema = super().compute_output_schema(input_schema, col_selector, prev_output_schema)

        adjusted_schema = Schema()
        for col_name, col_schema in schema.column_schemas.items():
            col_name = (
                f"{self.unrolled_prefix}_{col_name}"
                if col_name in self.unroll_cols and self.unrolled_prefix
                else col_name
            )
            adjusted_schema[col_name] = col_schema.with_name(col_name)

        return adjusted_schema

    def transform(self, df: InferenceDataFrame):
        num_items = df[self.candidate_col].shape[0]
        outputs = {}
        for col_name, col_value in df.tensors.items():
            outputs[col_name] = col_value

        for col in self.unroll_cols:
            target = outputs.pop(col)
            col_name = f"{self.unrolled_prefix}_{col}" if self.unrolled_prefix else col
            outputs[col_name] = cp.repeat(target, num_items, axis=0)

        return InferenceDataFrame(outputs)
