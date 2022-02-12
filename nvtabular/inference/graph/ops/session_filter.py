import json
import logging

import numpy as np

from nvtabular import ColumnSelector
from nvtabular.graph.schema import Schema
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator

LOG = logging.getLogger("nvt")


class FilterCandidates(PipelineableInferenceOperator):
    def __init__(self, candidate_col, filter_col):
        self.candidate_col = candidate_col
        self.filter_col = filter_col

    def transform(self, df: InferenceDataFrame):
        candidate_ids = df[self.candidate_col]
        filter_ids = df[self.filter_col]

        filtered_results = np.array([candidate_ids[~np.isin(candidate_ids, filter_ids)]]).T
        return InferenceDataFrame({self.candidate_col: filtered_results})

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return Schema([input_schema[self.candidate_col]])

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        candidate_col = parameters["candidate_col"]
        filter_col = parameters["filter_col"]
        return FilterCandidates(candidate_col, filter_col)

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}
        self_params = {
            "candidate_col": self.candidate_col,
            "filter_col": self.filter_col,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)
