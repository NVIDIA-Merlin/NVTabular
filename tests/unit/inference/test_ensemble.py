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
from distutils.spawn import find_executable

import pytest

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

import nvtabular as nvt  # noqa
import nvtabular.ops as wf_ops  # noqa
from nvtabular.loader.tf_utils import configure_tensorflow  # noqa

# everything tensorflow related must be imported after this.
configure_tensorflow()
triton = pytest.importorskip("nvtabular.inference.triton")
ensemble = pytest.importorskip("nvtabular.inference.triton.ensemble")
from nvtabular.inference.graph.ensemble import Ensemble  # noqa
from nvtabular.inference.graph.ops.tensorflow import TensorflowOp  # noqa
from nvtabular.inference.graph.ops.workflow import WorkflowOp  # noqa
from tests.unit.test_triton_inference import run_triton_server  # noqa

tritonclient = pytest.importorskip("tritonclient")
import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa

grpcclient = pytest.importorskip("tritonclient.grpc")

TRITON_SERVER_PATH = find_executable("tritonserver")

tf = pytest.importorskip("tensorflow")  # noqa


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_config_verification(tmpdir, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [nvt.graph.tags.Tags.USER]
        )
    selector = nvt.graph.selector.ColumnSelector(["x", "y", "id"])

    workflow_ops = selector >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops["x_nvt"])
    workflow.fit(dataset)

    # Create Tensorflow Model
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="x_nvt", dtype=tf.int32, shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, name="output"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Creating Triton Ensemble
    triton_chain = selector >> WorkflowOp(workflow, name="workflow") >> TensorflowOp(model)
    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    triton_ens.export(str(tmpdir))
    config_path = tmpdir / "ensemble_model" / "config.pbtxt"

    # Checking Triton Ensemble Config
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == "ensemble_model"
        assert parsed.platform == "ensemble"
        assert hasattr(parsed, "ensemble_scheduling")

    df = nvt.dispatch._make_df({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "id": [7, 8, 9]})

    output_columns = triton_ens.graph.output_schema.column_names
    response = _run_ensemble_on_tritonserver(str(tmpdir), output_columns, df, triton_ens.name)
    assert len(response.as_numpy("output")) == df.shape[0]


def _run_ensemble_on_tritonserver(
    tmpdir,
    output_columns,
    df,
    model_name,
):
    inputs = triton.convert_df_to_triton_input(df.columns, df)
    outputs = [grpcclient.InferRequestedOutput(col) for col in output_columns]
    response = None
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

    return response
