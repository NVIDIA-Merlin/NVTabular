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

from merlin.dag import ColumnSelector
from merlin.schema import ColumnSchema, Schema

from nvtabular.inference.graph.ops.operator import InferenceOperator
from nvtabular.inference.triton.ensemble import export_tensorflow_model


class TensorflowOp(InferenceOperator):
    def __init__(self, model):
        super().__init__()

        self.model = model

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

        self.model_inputs = [col.name.split("/")[0] for col in inputs]
        self.model_outputs = [col.name.split("/")[0] for col in outputs]

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Create a directory inside supplied path based on our export name"""
        # TODO: refactor out the nodeid logic check
        node_name = f"{self.export_name}_{node_id}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        return export_tensorflow_model(self.model, node_name, node_export_path, version=version)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        in_schema = Schema()
        for col, input_col in zip(self.model_inputs, self.model.inputs):
            in_schema.column_schemas[col] = ColumnSchema(col, dtype=input_col.dtype.as_numpy_dtype)

        return in_schema

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        model_selector = ColumnSelector(self.model_inputs)
        self._validate_matching_cols(input_schema, model_selector, "computing input selector")

        return model_selector

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        out_schema = Schema()
        for col, output_col in zip(self.model_outputs, self.model.outputs):
            out_schema.column_schemas[col] = ColumnSchema(
                col, dtype=output_col.dtype.as_numpy_dtype
            )
        return out_schema
