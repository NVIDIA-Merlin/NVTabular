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
import os
import pathlib
import tempfile
from shutil import copytree

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf  # noqa
from google.protobuf import text_format  # noqa

import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa
from nvtabular.graph.schema import ColumnSchema, Schema  # noqa
from nvtabular.graph.selector import ColumnSelector  # noqa
from nvtabular.inference.graph.ops.operator import InferenceOperator  # noqa
from nvtabular.inference.triton.ensemble import _convert_dtype  # noqa


class PredictTensorflow(InferenceOperator):
    def __init__(self, model_path, custom_objects=None):
        custom_objects = custom_objects or {}
        self.model_path = model_path

        self.model = tf.keras.models.load_model(self.model_path, custom_objects=custom_objects)

        signatures = getattr(self.model, "signatures", {}) or {}
        default_signature = signatures.get("serving_default")
        if not default_signature:
            # roundtrip saved self.model to disk to generate signature if it doesn't exist

            with tempfile.TemporaryDirectory() as tmp_dir:
                tf_model_path = pathlib.Path(tmp_dir) / "model.savedmodel"
                self.model.save(tf_model_path, include_optimizer=False)
                reloaded = tf.keras.models.load_model(tf_model_path)
                default_signature = reloaded.signatures["serving_default"]

        inputs = list(default_signature.structured_input_signature[1].values())
        outputs = list(default_signature.structured_outputs.values())

        input_col_names = [col.name.split("/")[0] for col in inputs]
        output_col_names = [col.name.split("/")[0] for col in outputs]

        self.input_schema = Schema()
        for col, input_col in zip(input_col_names, inputs):
            self.input_schema.column_schemas[col] = ColumnSchema(
                col, dtype=input_col.dtype.as_numpy_dtype
            )

        self.output_schema = Schema()
        for col, output_col in zip(output_col_names, outputs):
            self.output_schema.column_schemas[col] = ColumnSchema(
                col, dtype=output_col.dtype.as_numpy_dtype
            )

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        """Create a directory inside supplied path based on our export name"""
        # TODO: refactor out the nodeid logic check
        node_name = f"{self.export_name}_{node_id}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        tf_model_path = pathlib.Path(node_export_path) / str(version)
        # tf_model_path.mkdir(parents=True, exist_ok=True)
        # self.model.save(tf_model_path, include_optimizer=False)

        copytree(
            self.model_path,
            pathlib.Path(tf_model_path) / "model.savedmodel",
            dirs_exist_ok=True,
        )

        return export_tensorflow_model(self.model, node_name, node_export_path, version=version)

    def compute_input_schema(
        self,
        root_schema: Schema,
        parents_schema: Schema,
        deps_schema: Schema,
        selector: ColumnSelector,
    ) -> Schema:
        return self.input_schema

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        model_selector = ColumnSelector(self.input_schema.column_names)
        self._validate_matching_cols(input_schema, model_selector, "computing input selector")

        return model_selector

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        return self.output_schema


def export_tensorflow_model(model, name, output_path, version=1):
    """Exports a TensorFlow model for serving with Triton

    Parameters
    ----------
    model:
        The tensorflow model that should be served
    name:
        The name of the triton model to export
    output_path:
        The path to write the exported model to
    """
    tf_model_path = os.path.join(output_path, str(version), "model.savedmodel")
    # model.save(tf_model_path, include_optimizer=False)
    config = model_config.ModelConfig(
        name=name, backend="tensorflow", platform="tensorflow_savedmodel"
    )

    inputs, outputs = model.inputs, model.outputs

    if not inputs or not outputs:
        signatures = getattr(model, "signatures", {}) or {}
        default_signature = signatures.get("serving_default")
        if not default_signature:
            # roundtrip saved model to disk to generate signature if it doesn't exist

            reloaded = tf.keras.models.load_model(tf_model_path)
            default_signature = reloaded.signatures["serving_default"]

        inputs = list(default_signature.structured_input_signature[1].values())
        outputs = list(default_signature.structured_outputs.values())

    config.parameters["TF_GRAPH_TAG"].string_value = "serve"
    config.parameters["TF_SIGNATURE_DEF"].string_value = "serving_default"

    for col in inputs:
        config.input.append(
            model_config.ModelInput(
                name=f"{col.name}", data_type=_convert_dtype(col.dtype), dims=[-1, col.shape[1]]
            )
        )

    for col in outputs:
        # this assumes the list columns are 1D tensors both for cats and conts
        config.output.append(
            model_config.ModelOutput(
                name=col.name.split("/")[0],
                data_type=_convert_dtype(col.dtype),
                dims=[-1, col.shape[1]],
            )
        )

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config
