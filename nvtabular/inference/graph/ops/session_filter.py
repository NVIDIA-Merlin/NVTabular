import numpy as np

from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class FilterCandidates(PipelineableInferenceOperator):
    def __init__(self, candidate_col, filter_col):
        self.candidate_col = candidate_col
        self.filter_col = filter_col

    def transform(self, df: InferenceDataFrame):
        candidate_ids = df[self.candidate_col].as_numpy()
        filter_ids = df[self.filter_col].as_numpy()

        filtered_results = np.array([candidate_ids[~np.isin(candidate_ids, filter_ids)]]).T

        return InferenceDataFrame({self.candidate_col: filtered_results})
