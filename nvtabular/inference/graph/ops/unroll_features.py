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

import numpy as np

from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class UnrollFeatures(PipelineableInferenceOperator):
    def __init__(self, candidate_col, unroll_cols):
        self.candidate_col = candidate_col
        self.unroll_cols = unroll_cols

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        candidate_col = parameters["candidate_col"]
        unroll_cols = parameters["unroll_cols"]
        return UnrollFeatures(candidate_col, unroll_cols)

    def transform(self, df: InferenceDataFrame):
        num_items = df[self.candidate_col].shape[0]
        outputs = {}
        for col in self.cols:
            target = df[col].as_numpy
            outputs[col] = np.repeat(target, num_items, axis=0)

        return InferenceDataFrame(outputs)
