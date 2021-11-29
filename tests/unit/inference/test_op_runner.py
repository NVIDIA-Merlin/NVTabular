import json

import pytest

import nvtabular as nvt
import nvtabular.ops as wf_ops
from nvtabular.inference.graph.op_runner import OperatorRunner
from nvtabular.inference.graph.ops.workflow import WorkflowOp


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_loads_config(tmpdir, dataset, engine):
    input_columns = ["x", "y", "id"]

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops)
    workflow.fit(dataset)
    workflow.save(str(tmpdir))

    repository = "repository_path/"
    version = 1
    kind = ""
    config = {
        "parameters": {
            "operator_names": {"string_value": json.dumps(["WorkflowOp"])},
            "WorkflowOp": {
                "string_value": json.dumps(
                    {
                        "module_name": "nvtabular.inference.graph.ops.workflow",
                        "class_name": "WorkflowOp",
                        "workflow_path": str(tmpdir),
                    }
                )
            },
        }
    }

    runner = OperatorRunner(repository, version, kind, config)

    loaded_workflow_op = runner.operators[0]
    loaded_workflow = loaded_workflow_op.workflow
    assert isinstance(loaded_workflow_op, WorkflowOp)

    assert loaded_workflow.output_schema == workflow.output_schema
