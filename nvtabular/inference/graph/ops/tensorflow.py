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
import tempfile
from sys import version

from nvtabular.graph.schema import ColumnSchema, Schema
from nvtabular.graph.selector import ColumnSelector
from nvtabular.inference.graph.ops.operator import InferenceOperator
from nvtabular.inference.triton.ensemble import export_tensorflow_model


class TensorflowOp(InferenceOperator):
    def __init__(self, model, name=None):
        self.model = model
        self.name = name or self.__class__.__name__.lower()

    @property
    def export_name(self):
        return self.name

    def export(self, path, consumer_config, version=1):
        """Create a directory inside supplied path based on our export name"""
        new_dir_path = pathlib.Path(path) / self.export_name
        new_dir_path.mkdir()

        return export_tensorflow_model(self.model, self.export_name, new_dir_path, version=version)

    def compute_output_schema(self, input_schema: Schema, col_selector: ColumnSelector) -> Schema:
        expected_input = input_schema.apply(col_selector)
        inputs, outputs = self.model.inputs, self.model.outputs

        if not inputs or not outputs:
            signatures = getattr(self.model, "signatures", {}) or {}
            default_signature = signatures.get("serving_default")
            if not default_signature:
                # roundtrip saved self.model to disk to generate signature if it doesn't exist
                import tensorflow as tf

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tf_model_path = pathlib.Path(tmp_dir) / str(version) / "model.savedmodel"
                    self.model.save(tf_model_path, include_optimizer=False)
                    reloaded = tf.keras.self.models.load_model(tf_model_path)
                    default_signature = reloaded.signatures["serving_default"]

            inputs = list(default_signature.structured_input_signature[1].values())
            outputs = list(default_signature.structured_outputs.values())
        inputs = [col.name.split("/")[0] for col in inputs]
        outputs = [col.name.split("/")[0] for col in outputs]

        if expected_input.column_names != inputs:
            raise ValueError(
                f"Request schema provided to {self.__class__.__name__} \n"
                "doesn't match model's input schema.\n"
                f"Request schema columns: {expected_input.column_names}\n"
                f"Model input columns: {inputs}."
            )

        out_schema = Schema()
        for col in outputs:
            out_schema.column_schemas[col] = ColumnSchema(col)
        return out_schema
