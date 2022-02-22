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
from merlin.core.dispatch import make_df  # noqa
from merlin.schema import Tags  # noqa

import nvtabular as nvt  # noqa
import nvtabular.ops as wf_ops  # noqa

loader_tf_utils = pytest.importorskip("nvtabular.loader.tf_utils")

# everything tensorflow related must be imported after this.
loader_tf_utils.configure_tensorflow()
tf = pytest.importorskip("tensorflow")

triton = pytest.importorskip("nvtabular.inference.triton")
ensemble = pytest.importorskip("nvtabular.inference.triton.ensemble")
from nvtabular.inference.graph.ensemble import Ensemble  # noqa
from nvtabular.inference.graph.ops.tensorflow import TensorflowOp  # noqa
from nvtabular.inference.graph.ops.workflow import WorkflowOp  # noqa
from tests.unit.inference.inf_test_ops import PlusTwoOp  # noqa
from tests.unit.inference.inference_utils import (  # noqa
    _run_ensemble_on_tritonserver,
    create_tf_model,
)

tritonclient = pytest.importorskip("tritonclient")
import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa

grpcclient = pytest.importorskip("tritonclient.grpc")

TRITON_SERVER_PATH = find_executable("tritonserver")


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_config_verification(tmpdir, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )
    selector = nvt.graph.selector.ColumnSelector(["x", "y", "id"])

    workflow_ops = selector >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops["x_nvt"])
    workflow.fit(dataset)

    # Create Tensorflow Model
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="x_nvt", dtype=tf.float64, shape=(1,)),
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
    triton_chain = selector >> WorkflowOp(workflow, cats=["x_nvt"]) >> TensorflowOp(model)
    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, node_configs = triton_ens.export(str(tmpdir))

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

    df = make_df({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0], "id": [7, 8, 9]})

    output_columns = triton_ens.graph.output_schema.column_names
    response = _run_ensemble_on_tritonserver(str(tmpdir), output_columns, df, triton_ens.name)
    assert len(response.as_numpy("output")) == df.shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_multi_op_run(tmpdir, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )

    workflow_ops = ["name-cat"] >> nvt.ops.Categorify(cat_cache="host")
    workflow = nvt.Workflow(workflow_ops)
    workflow.fit(dataset)

    embedding_shapes_1 = nvt.ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> nvt.ops.Categorify(cat_cache="host")
    workflow_2 = nvt.Workflow(cats)
    workflow_2.fit(dataset)

    embedding_shapes = nvt.ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)
    # Create Tensorflow Model
    model = create_tf_model(["name-cat", "name-string"], [], embedding_shapes_1)

    # Creating Triton Ensemble
    triton_chain_1 = ["name-cat"] >> WorkflowOp(workflow)
    triton_chain_2 = ["name-string"] >> WorkflowOp(workflow_2)
    triton_chain = (triton_chain_1 + triton_chain_2) >> TensorflowOp(model)

    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, nodes_config = triton_ens.export(str(tmpdir))
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

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]

    response = _run_ensemble_on_tritonserver(str(tmpdir), ["output"], df, triton_ens.name)
    assert len(response.as_numpy("output")) == df.shape[0]


@pytest.mark.skipif(not TRITON_SERVER_PATH, reason="triton server not found")
@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_tf_e2e_multi_op_plus_2_run(tmpdir, dataset, engine):
    # Create a Workflow
    schema = dataset.schema
    for name in ["x", "y", "id"]:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [Tags.USER]
        )

    workflow_ops = ["name-cat"] >> nvt.ops.Categorify(cat_cache="host")
    workflow = nvt.Workflow(workflow_ops)
    workflow.fit(dataset)

    embedding_shapes_1 = nvt.ops.get_embedding_sizes(workflow)

    cats = ["name-string"] >> nvt.ops.Categorify(cat_cache="host")
    workflow_2 = nvt.Workflow(cats)
    workflow_2.fit(dataset)

    embedding_shapes = nvt.ops.get_embedding_sizes(workflow_2)
    embedding_shapes_1.update(embedding_shapes)
    embedding_shapes_1["name-string_plus_2"] = embedding_shapes_1["name-string"]

    # Create Tensorflow Model
    model = create_tf_model(["name-cat", "name-string_plus_2"], [], embedding_shapes_1)

    # Creating Triton Ensemble
    triton_chain_1 = ["name-cat"] >> WorkflowOp(workflow)
    triton_chain_2 = ["name-string"] >> WorkflowOp(workflow_2) >> PlusTwoOp()
    triton_chain = (triton_chain_1 + triton_chain_2) >> TensorflowOp(model)

    triton_ens = Ensemble(triton_chain, schema)

    # Creating Triton Ensemble Config
    ensemble_config, nodes_config = triton_ens.export(str(tmpdir))
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

    df = dataset.to_ddf().compute()[["name-string", "name-cat"]].iloc[:3]

    response = _run_ensemble_on_tritonserver(str(tmpdir), ["output"], df, triton_ens.name)
    assert len(response.as_numpy("output")) == df.shape[0]
