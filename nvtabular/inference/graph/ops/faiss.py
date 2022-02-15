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

import cupy as cp
import faiss
import numpy as np

from nvtabular import ColumnSchema, ColumnSelector
from nvtabular.graph.schema import Schema
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class QueryFaiss(PipelineableInferenceOperator):
    def __init__(self, index_path, query_vector_col="query_vector", topk=10):
        self.index_path = str(index_path)
        self.index = faiss.read_index(str(index_path))
        self.query_vector_col = query_vector_col
        self.topk = topk

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        index_path = parameters["index_path"]
        topk = parameters["topk"]
        query_vector_col = parameters["query_vector_col"]

        return QueryFaiss(index_path, query_vector_col=query_vector_col, topk=topk)

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}
        self_params = {
            "index_path": self.index_path,
            "topk": self.topk,
            "query_vector_col": self.query_vector_col,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(self, df: InferenceDataFrame):
        user_vector = df[self.query_vector_col]

        _, indices = self.index.search(user_vector, self.topk)
        # distances, indices = self.index.search(user_vector, self.topk)

        candidate_ids = cp.array(indices).T.astype(np.int32)
        # candidate_distances = np.array(distances).T.astype(np.float32)

        return InferenceDataFrame(
            # {"candidate_ids": candidate_ids, "candidate_distances": candidate_distances}
            {"candidate_ids": candidate_ids}
        )

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return Schema(
            [
                ColumnSchema("candidate_ids", dtype=np.int32),
                # ColumnSchema("candidate_distances", dtype=np.float32),
            ]
        )


def setup_faiss(item_vector, output_path):
    index = faiss.IndexFlatL2(item_vector[0].shape[0])
    index.add(item_vector)
    faiss.write_index(index, str(output_path))
