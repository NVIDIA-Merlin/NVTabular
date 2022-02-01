# import json

import numpy as np
from pymilvus import (  # utility,     <-- useful for dropping the collection
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
)

from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class QueryMilvus(PipelineableInferenceOperator):
    def __init__(self, host, port, milvus_collection):
        self.milvus_client = connections.connect(host=host, port=port)
        self.milvus_collection = milvus_collection
        self.milvus_collection.load()

    @classmethod
    def from_config(cls, config):
        # model_config = json.loads(config["model_config"])

        # TODO: Somehow, load the collection schema in here
        host = "milvus-standalone"
        port = "19530"
        dim = 128
        default_fields = [
            FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="item_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        default_schema = CollectionSchema(
            fields=default_fields, description="MovieLens item vectors"
        )
        milvus_collection = Collection(
            name="movielens_retrieval_tf", data=None, schema=default_schema
        )

        return QueryMilvus(host, port, milvus_collection)

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        # TODO: Somehow, store the collection schema in here
        self_params = {}
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(self, df: InferenceDataFrame):
        user_vector = df["user_vector"].as_numpy()

        # Normalize the user query vector
        user_vector = user_vector / np.sqrt(np.sum(user_vector ** 2))

        topK = 100
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        result = self.milvus_collection.search(user_vector, "item_vector", search_params, topK)

        candidate_ids = np.array([[h.id for hits in result for h in hits]]).T
        candidate_distances = np.array([[h.distance for hits in result for h in hits]]).T

        return InferenceDataFrame(
            {"candidate_ids": candidate_ids, "candidate_distances": candidate_distances}
        )
