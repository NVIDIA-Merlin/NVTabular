import numpy as np
from feast import FeatureStore

from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator


class QueryFeast(PipelineableInferenceOperator):
    def __init__(self, repo_path, entity_id, entity_view, features, mh_features):

        self.store = FeatureStore(repo_path=repo_path)
        self.entity_id = entity_id
        self.entity_view = entity_view
        self.features = features
        self.mh_features = mh_features

    @classmethod
    def from_config(cls, config):
        parameters = config["parameters"]["string_value"]
        entity_id = parameters["entity_id"]["string_value"]
        entity_view = parameters["entity_view"]["string_value"]
        features = parameters["features"]["string_value"]
        mh_features = parameters["feature"]["string_value"]
        repo_path = parameters["feast_repo_path"]["string_value"]

        return QueryFeast(repo_path, entity_id, entity_view, features, mh_features)

    def export(self, path, input_schema, output_schema, params=None, node_id=None, version=1):
        self_params = {
            "entity_id": self.entity_id,
            "entity_view": self.entity_view,
            "features": self.features,
            "mh_features": self.mh_features,
        }
        self_params.update(params)
        return super().export(path, input_schema, output_schema, self_params, node_id, version)

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        entity_ids = df[self.entity_id]
        entity_rows = [{self.entity_id: int(entity_id)} for entity_id in entity_ids]

        feature_names = self.features + self.mh_features
        feature_refs = [
            ":".join([self.entity_view, feature_name]) for feature_name in feature_names
        ]

        features = self.store.get_online_features(
            feature_refs=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        output_tensors = {}

        # Numerical and single-hot categorical
        for feature_name in self.features:
            feature_value = features[f"{self.entity_view}__{feature_name}"]
            feature_array = np.array([feature_value]).T
            output_tensors[feature_name] = feature_array

        # Multi-hot categorical
        for feature_name in self.mh_features:
            feature_value = features[f"{self.entity_view}__{feature_name}"]
            feature_array = np.array(feature_value).T

            output_tensors[f"{feature_name}__values"] = feature_array
            output_tensors[f"{feature_name}__nnzs"] = np.array(
                [[len(feature_array)]], dtype=np.int32
            )

        return InferenceDataFrame(output_tensors)
