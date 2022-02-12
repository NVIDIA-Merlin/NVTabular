import json
import logging

import numpy as np
from feast import FeatureStore

from nvtabular import ColumnSchema
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator

LOG = logging.getLogger("nvt")


class QueryFeast(PipelineableInferenceOperator):
    def __init__(
        self,
        repo_path,
        entity_id,
        entity_view,
        entity_column,
        features,
        mh_features,
        input_schema,
        output_schema,
    ):

        self.repo_path = repo_path
        self.store = FeatureStore(repo_path=repo_path)
        self.entity_id = entity_id
        self.entity_view = entity_view
        self.entity_column = entity_column
        self.features = features
        self.mh_features = mh_features
        self.input_schema = input_schema
        self.output_schema = output_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return self.output_schema

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        return self.input_schema

    @classmethod
    def from_config(cls, config):
        parameters = json.loads(config.get("params", ""))
        entity_id = parameters["entity_id"]
        entity_view = parameters["entity_view"]
        entity_column = parameters["entity_column"]
        features = parameters["features"]
        mh_features = parameters["mh_features"]
        repo_path = parameters["feast_repo_path"]
        in_dict = json.loads(config.get("input_dict", "{}"))
        out_dict = json.loads(config.get("output_dict", "{}"))

        in_schema = Schema([])
        for col_name, col_rep in in_dict.items():
            in_schema[col_name] = ColumnSchema(
                col_name, dtype=col_rep["dtype"], _is_list=col_rep["is_list"]
            )
        out_schema = Schema([])
        for col_name, col_rep in out_dict.items():
            out_schema[col_name] = ColumnSchema(
                col_name, dtype=col_rep["dtype"], _is_list=col_rep["is_list"]
            )

        return QueryFeast(
            repo_path,
            entity_id,
            entity_view,
            entity_column,
            features,
            mh_features,
            in_schema,
            out_schema,
        )

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        params = params or {}
        self_params = {
            "entity_id": self.entity_id,
            "entity_view": self.entity_view,
            "entity_column": self.entity_column,
            "features": self.features,
            "mh_features": self.mh_features,
            "feast_repo_path": self.repo_path,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        entity_ids = df[self.entity_column]
        entity_rows = [{self.entity_id: int(entity_id)} for entity_id in entity_ids]

        feature_names = self.features + self.mh_features
        feature_refs = [
            ":".join([self.entity_view, feature_name]) for feature_name in feature_names
        ]

        feast_response = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()
        output_tensors = {}

        # Numerical and single-hot categorical
        for feature_name in self.features:
            feature_value = feast_response[feature_name]
            feature_array = np.array([feature_value]).T.astype(
                self.output_schema[feature_name].dtype
            )
            output_tensors[feature_name] = feature_array

        # Multi-hot categorical
        for feature_name in self.mh_features:
            feature_value = feast_response[feature_name]
            feature_array = np.array(feature_value).T.astype(
                self.output_schema[f"{feature_name}_1"].dtype
            )

            feature_nnzs = np.array(
                [[len(feature_array)]], dtype=self.output_schema[f"{feature_name}_2"].dtype
            )

            output_tensors[f"{feature_name}_1"] = feature_array
            output_tensors[f"{feature_name}_2"] = feature_nnzs

        return InferenceDataFrame(output_tensors)
