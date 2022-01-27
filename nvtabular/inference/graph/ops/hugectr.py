#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json
import os
import pathlib
from shutil import copyfile

import numpy as np

from nvtabular.graph.schema import ColumnSchema, Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.graph.tags import Tags
from nvtabular.inference.graph.ops.operator import (
    InferenceDataFrame,
    InferenceOperator,
    PipelineableInferenceOperator,
    _schema_to_dict,
)
from nvtabular.inference.triton.ensemble import _convert_dtype, generate_hugectr_model

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa


class HugeCTROp(InferenceOperator):
    def __init__(self, model_path, params=None, name=None, max_batch_size=None, label_columns=None):
        self.model_path = model_path
        self.name = name or self.__class__.__name__.lower()
        self.params = params or {}
        self.max_batch_size = max_batch_size or 0
        self.label_columns = label_columns or []

        self.params["label_dim"] = len(self.label_columns)

    @property
    def export_name(self):
        return self.name

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        input_schema = super().compute_input_schema(
            root_schema, parents_schema, deps_schema, selector
        )
        self.params["des_feature_num"] = len(input_schema.select_by_tag(Tags.CONTINUOUS))
        self.params["cat_feature_num"] = len(input_schema.select_by_tag(Tags.CATEGORICAL))

        return Schema(
            [
                ColumnSchema("DES", dtype=np.float32),
                ColumnSchema("CATCOLUMN", dtype=np.uint32),
                ColumnSchema("ROWINDEX", dtype=np.int32),
            ]
        )

    # ensemble_conf, nvt_hugectr_conf = export_hugectr_ensemble(
    #     workflow=workflow,
    #     hugectr_model_path=str(TRAIN_DIR),
    #     hugectr_params=hugectr_params,
    #     name=MODEL_NAME,
    #     output_path=str(MODEL_DIR),
    #     label_columns=["rating"],
    #     cats=CATEGORICAL_COLUMNS,
    #     max_batch_size=64,
    # )
    def export(self, path, input_schema, output_schema, version=1):
        """Create a directory inside supplied path based on our export name"""
        new_dir_path = pathlib.Path(path)
        new_dir_path.mkdir(exist_ok=True)

        return generate_hugectr_model(
            self.model_path,
            self.params,
            self.export_name,
            new_dir_path,
            version=version,
            max_batch_size=self.max_batch_size,
        )

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        # for col in self.label_columns:
        #     input_schema.remove_col(col)
        # expected_input = input_schema.apply(col_selector)
        # # compute inputs from params, cannot retrieve
        # # maybe get amount of columns that should exist?
        # len_inputs = self.params["slots"]
        # out_schema = Schema()

        # for col_num in range(self.params["n_outputs"]):
        #     col = f"OUTPUT{col_num}"
        #     out_schema.column_schemas[col] = ColumnSchema(col, dtype=np.dtype(np.float32))

        # # if len(expected_input.column_names) != len_inputs:
        # #     raise ValueError(
        # #         f"Request schema provided to {self.__class__.__name__} \n"
        # #         "doesn't match model's input schema.\n"
        # #         f"Request schema columns: {len(expected_input.column_names)}\n"
        # #         f"Model input columns: {len_inputs}."
        # #     )
        return Schema([ColumnSchema("OUTPUT0", dtype=np.float32)])


class HugeCTRSetOp(PipelineableInferenceOperator):
    @property
    def export_name(self):
        return str(self.__class__.__name__).lower()

    @classmethod
    def from_config(cls, config):
        return HugeCTRSetOp

    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:

        return df

    def export(self, export_path, input_schema, output_schema, version=1):
        full_export_path = pathlib.Path(export_path) / self.export_name
        full_export_path.mkdir(exist_ok=True)
        config = model_config.ModelConfig(
            name=self.export_name, backend="nvtabular", platform="op_runner"
        )

        config.parameters["operator_names"].string_value = json.dumps([self.__class__.__name__])

        config.parameters[self.__class__.__name__].string_value = json.dumps(
            {
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                "input_dict": json.dumps(_schema_to_dict(input_schema)),
                "output_dict": json.dumps(_schema_to_dict(output_schema)),
            }
        )

        for col_name, col_dict in _schema_to_dict(input_schema).items():
            config.input.append(
                model_config.ModelInput(
                    name=col_name, data_type=_convert_dtype(col_dict["dtype"]), dims=[-1, 1]
                )
            )

        config.output.append(
            model_config.ModelOutput(name="DES", data_type=model_config.TYPE_FP32, dims=[-1])
        )

        config.output.append(
            model_config.ModelOutput(name="CATCOLUMN", data_type=model_config.TYPE_INT64, dims=[-1])
        )

        config.output.append(
            model_config.ModelOutput(name="ROWINDEX", data_type=model_config.TYPE_INT32, dims=[-1])
        )

        with open(os.path.join(full_export_path, "config.pbtxt"), "w") as o:
            text_format.PrintMessage(config, o)

        os.makedirs(full_export_path, exist_ok=True)
        os.makedirs(os.path.join(full_export_path, str(version)), exist_ok=True)
        copyfile(
            os.path.join(os.path.dirname(__file__), "..", "..", "triton", "oprunner_model.py"),
            os.path.join(full_export_path, str(version), "model.py"),
        )

        return config

    def column_mapping(self, col_selector):
        cats = self.cats or []
        conts = self.conts or []

        return {
            "DES": conts,
            "CATCOLUMN": cats,
            "ROWINDEX": [],
        }

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        input_schema = super().compute_input_schema(
            root_schema, parents_schema, deps_schema, selector
        )
        self.cats = input_schema.select_by_tag(Tags.CONTINUOUS)
        self.conts = input_schema.select_by_tag(Tags.CATEGORICAL)

        return input_schema
