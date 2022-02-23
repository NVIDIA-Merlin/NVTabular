import json
import logging

import numpy as np
from feast import FeatureStore, ValueType

from nvtabular import ColumnSchema
from nvtabular.graph.schema import Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.inference.graph.ops.operator import InferenceDataFrame, PipelineableInferenceOperator

LOG = logging.getLogger("nvt")

# Feast_key: (numpy dtype, is_list, is_ragged)
feast_2_numpy = {
    ValueType.INT64: (np.int64, False, False),
    ValueType.INT32: (np.int32, False, False),
    ValueType.FLOAT: (np.float, False, False),
    ValueType.INT64_LIST: (np.int64, True, True),
    ValueType.INT32_LIST: (np.int32, True, True),
    ValueType.FLOAT_LIST: (np.float, True, True),
}


class QueryFeast(PipelineableInferenceOperator):
    @classmethod
    def from_feature_view(cls, store, path, view, column, output_prefix=None, include_id=False):
        feature_view = store.get_feature_view(view)
        entity_id = feature_view.entities[0]

        features = []
        mh_features = []

        input_schema = Schema([ColumnSchema(column, dtype=np.int32)])

        output_schema = Schema([])
        for feature in feature_view.features:
            feature_dtype, is_list, is_ragged = feast_2_numpy[feature.dtype]

            if is_list:
                mh_features.append(feature.name)

                values_name = cls._prefixed_name(output_prefix, f"{feature.name}_1")
                nnzs_name = cls._prefixed_name(output_prefix, f"{feature.name}_2")
                output_schema[values_name] = ColumnSchema(
                    values_name, dtype=feature_dtype, _is_list=is_list, _is_ragged=is_ragged
                )
                output_schema[nnzs_name] = ColumnSchema(
                    nnzs_name, dtype=np.int64, _is_list=True, _is_ragged=False
                )
            else:
                features.append(feature.name)

                name = cls._prefixed_name(output_prefix, feature.name)
                output_schema[name] = ColumnSchema(
                    name, dtype=feature_dtype, _is_list=is_list, _is_ragged=is_ragged
                )

        if include_id:
            output_schema[entity_id] = ColumnSchema(entity_id, dtype=np.int32)

        return QueryFeast(
            path,
            entity_id,
            view,
            column,
            features,
            mh_features,
            input_schema,
            output_schema,
            include_id=include_id,
            output_prefix=output_prefix or "",
            suffix_int=1,
        )

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
        include_id=False,
        output_prefix="",
        suffix_int=1,
    ):
        self.repo_path = repo_path
        self.entity_id = entity_id
        self.entity_view = entity_view
        self.entity_column = entity_column

        self.features = features
        self.mh_features = mh_features
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.include_id = include_id
        self.output_prefix = output_prefix
        self.suffix_int = suffix_int

        self.store = FeatureStore(repo_path=repo_path)

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
        repo_path = parameters["feast_repo_path"]
        features = parameters["features"]
        mh_features = parameters["mh_features"]
        in_dict = json.loads(config.get("input_dict", "{}"))
        out_dict = json.loads(config.get("output_dict", "{}"))
        include_id = parameters["include_id"]
        output_prefix = parameters["output_prefix"]
        suffix_int = parameters["suffix_int"]

        in_schema = Schema([])
        for col_name, col_rep in in_dict.items():
            in_schema[col_name] = ColumnSchema(
                col_name,
                dtype=col_rep["dtype"],
                _is_list=col_rep["is_list"],
                _is_ragged=col_rep["is_ragged"],
            )
        out_schema = Schema([])
        for col_name, col_rep in out_dict.items():
            out_schema[col_name] = ColumnSchema(
                col_name,
                dtype=col_rep["dtype"],
                _is_list=col_rep["is_list"],
                _is_ragged=col_rep["is_ragged"],
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
            include_id,
            output_prefix,
            suffix_int,
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
            "include_id": self.include_id,
            "output_prefix": self.output_prefix,
            "suffix_int": self.suffix_int,
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
        if self.include_id:
            output_tensors[self.entity_id] = entity_ids

        # Numerical and single-hot categorical
        for feature_name in self.features:
            prefixed_name = self.__class__._prefixed_name(self.output_prefix, feature_name)

            feature_value = feast_response[feature_name]
            feature_array = np.array([feature_value]).T.astype(
                self.output_schema[prefixed_name].dtype
            )
            output_tensors[prefixed_name] = feature_array

        # Multi-hot categorical
        for feature_name in self.mh_features:
            feature_value = feast_response[feature_name]

            prefixed_name = self.__class__._prefixed_name(self.output_prefix, feature_name)
            feature_out_name = f"{prefixed_name}_{self.suffix_int}"

            nnzs = None
            if (
                isinstance(feature_value[0], list)
                and self.output_schema[feature_out_name]._is_ragged
            ):
                flattened_value = []
                for val in feature_value:
                    flattened_value.extend(val)

                nnzs = [len(vals) for vals in feature_value]
                feature_value = [flattened_value]

            feature_array = np.array(feature_value).T.astype(
                self.output_schema[feature_out_name].dtype
            )
            if not nnzs:
                nnzs = [len(feature_array)]
            feature_out_nnz = f"{prefixed_name}_{self.suffix_int+1}"
            feature_nnzs = np.array([nnzs], dtype=self.output_schema[feature_out_nnz].dtype).T

            output_tensors[feature_out_name] = feature_array
            output_tensors[feature_out_nnz] = feature_nnzs

        return InferenceDataFrame(output_tensors)

    @classmethod
    def _prefixed_name(cls, output_prefix, col_name):
        if output_prefix and col_name and not col_name.startswith(output_prefix):
            return f"{output_prefix}_{col_name}"
        else:
            return col_name
