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
import pathlib

import numpy as np

from nvtabular.graph.schema import ColumnSchema, Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.graph.tags import Tags
from nvtabular.inference.graph.ops.operator import InferenceOperator
from nvtabular.inference.triton.ensemble import generate_hugectr_model


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
        return input_schema

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        for col in self.label_columns:
            input_schema.remove_col(col)
        expected_input = input_schema.apply(col_selector)
        # compute inputs from params, cannot retrieve
        # maybe get amount of columns that should exist?
        len_inputs = self.params["slots"]
        outputs = []
        out_schema = Schema()

        for col_num in range(self.params["n_outputs"]):
            col = f"OUTPUT{col_num}"
            out_schema.column_schemas[col] = ColumnSchema(col, dtype=np.dtype(np.float32))

        if len(expected_input.column_names) != len_inputs:
            raise ValueError(
                f"Request schema provided to {self.__class__.__name__} \n"
                "doesn't match model's input schema.\n"
                f"Request schema columns: {len(expected_input.column_names)}\n"
                f"Model input columns: {len_inputs}."
            )

        out_schema = Schema()
        for col in outputs:
            out_schema.column_schemas[col] = ColumnSchema(col)
        return out_schema
