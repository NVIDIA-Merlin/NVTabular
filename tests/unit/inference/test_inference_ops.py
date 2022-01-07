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

import pytest

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

import nvtabular as nvt  # noqa
import nvtabular.ops as wf_ops  # noqa
from nvtabular.graph.schema import Schema  # noqa

ensemble = pytest.importorskip("nvtabular.inference.graph.ensemble")
model_config = pytest.importorskip("nvtabular.inference.triton.model_config_pb2")
workflow_op = pytest.importorskip("nvtabular.inference.graph.ops.workflow")


@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_op_validates_schemas(dataset, engine):
    input_columns = ["x", "y", "id"]
    request_schema = Schema(input_columns)

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops)
    workflow.fit(dataset)

    # Triton
    triton_ops = ["a", "b", "c"] >> workflow_op.WorkflowOp(workflow)

    with pytest.raises(ValueError) as exc_info:
        ensemble.Ensemble(triton_ops, request_schema)
        assert "Missing column" in str(exc_info.value)


@pytest.mark.parametrize("engine", ["parquet"])
def test_workflow_op_exports_own_config(tmpdir, dataset, engine):
    input_columns = ["x", "y", "id"]

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops)
    workflow.fit(dataset)

    # Triton
    triton_op = workflow_op.WorkflowOp(workflow, name="workflow")
    triton_op.export(tmpdir)

    # Export creates directory
    export_path = pathlib.Path(tmpdir) / triton_op.export_name
    assert export_path.exists()

    # Export creates the config file
    config_path = export_path / "config.pbtxt"
    assert config_path.exists()

    # Read the config file back in from proto
    with open(config_path, "rb") as f:
        config = model_config.ModelConfig()
        raw_config = f.read()
        parsed = text_format.Parse(raw_config, config)

        # The config file contents are correct
        assert parsed.name == "workflow"
        assert parsed.backend == "nvtabular"
