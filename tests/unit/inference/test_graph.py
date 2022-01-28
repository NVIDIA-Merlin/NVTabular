import pytest

import nvtabular as nvt
import nvtabular.ops as wf_ops
from nvtabular.graph.schema import Schema

ensemble = pytest.importorskip("nvtabular.inference.graph.ensemble")
workflow_op = pytest.importorskip("nvtabular.inference.graph.ops.workflow")


def test_inference_schema_propagation():
    input_columns = ["a", "b", "c"]
    request_schema = Schema(input_columns)
    expected_schema = Schema(["a_nvt", "b_nvt", "c_nvt"])

    # NVT
    workflow_ops = input_columns >> wf_ops.Rename(postfix="_nvt")
    workflow = nvt.Workflow(workflow_ops)
    workflow.fit_schema(request_schema)

    assert workflow.graph.output_schema == expected_schema

    # Triton
    triton_ops = input_columns >> workflow_op.WorkflowOp(workflow)
    ensemble_out = ensemble.Ensemble(triton_ops, request_schema)

    assert ensemble_out.graph.output_schema == expected_schema
