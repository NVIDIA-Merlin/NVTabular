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

import faiss
import numpy as np

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema
from merlin.systems.dag.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class QueryFaiss(PipelineableInferenceOperator):
    def __init__(self, index_path, topk=10):
        self.index_path = str(index_path)
        self.topk = topk
        self._index = None
        super().__init__()

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        index_path = parameters["index_path"]
        topk = parameters["topk"]

        operator = QueryFaiss(index_path, topk=topk)
        operator._index = faiss.read_index(str(index_path))

        return operator

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}

        # TODO: Copy the index into the export directory

        self_params = {
            # TODO: Write the (relative) path from inside the export directory
            "index_path": self.index_path,
            "topk": self.topk,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(self, df: InferenceDataFrame):
        user_vector = list(df.tensors.values())[0]

        _, indices = self._index.search(user_vector, self.topk)
        # distances, indices = self.index.search(user_vector, self.topk)

        candidate_ids = np.array(indices).T.astype(np.int32)

        return InferenceDataFrame({"candidate_ids": candidate_ids})

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
        if len(input_schema.column_schemas) > 1:
            raise ValueError(
                "More than one input has been detected for this node,"
                / f"inputs received: {input_schema.column_names}"
            )
        return input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return Schema(
            [
                ColumnSchema("candidate_ids", dtype=np.int32),
            ]
        )


def setup_faiss(item_vector, output_path):
    index = faiss.IndexFlatL2(item_vector[0].shape[0])
    index.add(item_vector)
    faiss.write_index(index, str(output_path))
