import json

import numpy as np

from nvtabular import ColumnSchema
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class SoftmaxSampling(PipelineableInferenceOperator):
    def __init__(self, candidate_col, relevance_col, temperature=20.0, topk=10):
        self.candidate_col = candidate_col
        self.relevance_col = relevance_col
        self.temperature = temperature
        self.topk = topk

    @classmethod
    def from_config(cls, config):
        """Load operator and properties from Triton config"""
        parameters = json.loads(config.get("params", ""))
        candidate_col = parameters["candidate_col"]
        predict_col = parameters["relevance_col"]
        topk = parameters["topk"]
        theta = parameters["temperature"]

        return SoftmaxSampling(candidate_col, predict_col, temperature=theta, topk=topk)

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        """Write out a Triton model config directory"""
        params = params or {}
        self_params = {
            "candidate_col": self.candidate_col,
            "relevance_col": self.relevance_col,
            "temperature": self.temperature,
            "topk": self.topk,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        """Describe the operator's outputs"""
        return Schema([ColumnSchema("ordered_ids", dtype=np.int32, _is_list=True, _is_ragged=True)])

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        """Transform the dataframe by applying this operator to the set of input columns"""
        # Extract parameters from the request
        candidate_ids = df[self.candidate_col].reshape(-1)

        predicted_scores = df[self.relevance_col].reshape(-1)

        # Exponential sort trick for sampling from a distribution without replacement from:

        # Pavlos S. Efraimidis, Paul G. Spirakis, Weighted random sampling with a reservoir,
        # Information Processing Letters, Volume 97, Issue 5, 2006, Pages 181-185, ISSN 0020-0190,
        # https://doi.org/10.1016/j.ipl.2005.11.003.

        # As implemented by Tim Vieira in "Algorithms for sampling without replacement"
        # https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/

        # The weights for the sampling distribution are the softmax of the scores
        weights = np.exp(self.temperature * predicted_scores) / np.sum(predicted_scores)

        # This is the core of the exponential sampling trick, which creates a
        # set of values that depend on both the predicted scores and random
        # variables, resulting in a set of values that will sort into an order
        # that reflects sampling without replacement according to the weight
        # distribution
        num_items = candidate_ids.shape[0]
        exponentials = -np.log(np.random.uniform(0, 1, size=(num_items,)))
        exponentials /= weights

        # This is just bookkeeping to produce the final ordered list of recs
        sorted_indices = np.argsort(exponentials)
        topk_movie_ids = candidate_ids[sorted_indices][: self.topk]
        ordered_movie_ids = topk_movie_ids.reshape(1, -1).T

        return InferenceDataFrame({"ordered_ids": ordered_movie_ids})
