import numpy as np

from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class UnrollFeatures(PipelineableInferenceOperator):
    def __init__(self, candidate_col, unroll_cols):
        self.candidate_col = candidate_col
        self.unroll_cols = unroll_cols

    def transform(self, df: InferenceDataFrame):
        num_items = df[self.candidate_col].shape[0]
        outputs = {}
        for col in self.cols:
            target = df[col].as_numpy
            outputs[col] = np.repeat(target, num_items, axis=0)

        return InferenceDataFrame(outputs)
