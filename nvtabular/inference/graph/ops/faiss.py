import json

import faiss
import numpy as np

from nvtabular import ColumnSchema, ColumnSelector
from nvtabular.graph.schema import Schema
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class QueryFaiss(PipelineableInferenceOperator):
    def __init__(self, index_path, topk=10):
        self.index_path = str(index_path)
        self.index = faiss.read_index(str(index_path))
        self.topk = topk

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        index_path = parameters["index_path"]
        topk = parameters["topk"]

        return QueryFaiss(index_path, topk=topk)

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}
        self_params = {"index_path": self.index_path, "topk": self.topk}
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(self, df: InferenceDataFrame):
        user_vector = df["output_1"]

        _, indices = self.index.search(user_vector, self.topk)
        # distances, indices = self.index.search(user_vector, self.topk)

        candidate_ids = np.array(indices).T.astype(np.int32)
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
    index = faiss.IndexFlatL2(len(item_vector[0]))
    index.add(item_vector)
    faiss.write_index(index, str(output_path))
