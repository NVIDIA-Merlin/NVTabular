import json

import numpy as np

from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class SoftMaxSampling(PipelineableInferenceOperator):
    def __init__(self, candidate_col, predict_col, k=None, theta=None):
        self.candidate_col = candidate_col
        self.predict_col = predict_col
        self.k = k or 10
        self.theta = theta or 20.0

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        candidate_col = parameters["candidate_col"]
        predict_col = parameters["predict_col"]
        k = parameters["k"]
        theta = parameters["theta"]

        SoftMaxSampling(candidate_col, predict_col, k=k, theta=theta)

    def execute(self, df: InferenceDataFrame):
        # Extract parameters from the request
        candidate_ids = df[self.candidate_col].as_numpy().reshape(-1)

        predicted_scores = df[self.predict_col].as_numpy().reshape(-1)

        # Exponential sort trick for sampling from a distribution without replacement from:

        # Pavlos S. Efraimidis, Paul G. Spirakis, Weighted random sampling with a reservoir,
        # Information Processing Letters, Volume 97, Issue 5, 2006, Pages 181-185, ISSN 0020-0190,
        # https://doi.org/10.1016/j.ipl.2005.11.003.

        # As implemented by Tim Vieira in "Algorithms for sampling without replacement"
        # https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/

        # TODO: Extract k and theta as config parameters
        k = 10
        theta = 20.0

        # The weights for the sampling distribution are the softmax of the scores
        weights = np.exp(theta * predicted_scores) / np.sum(predicted_scores)

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
        topk_movie_ids = candidate_ids[sorted_indices][:k]
        ordered_movie_ids = topk_movie_ids.reshape(1, -1).T

        return InferenceDataFrame({"ordered_ids": ordered_movie_ids})
