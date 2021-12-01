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


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_loads_multiple_ops_same(tmpdir, dataset, engine):
    # NVT
    schema = dataset.schema
    for name in schema.column_names:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [nvt.graph.tags.Tags.USER]
        )
    selector = nvt.graph.selector.ColumnSelector(tags=[nvt.graph.tags.Tags.USER])

    workflow_ops_1 = selector >> wf_ops.Rename(postfix="_1")
    workflow_1 = nvt.Workflow(workflow_ops_1)
    workflow_1.fit(dataset)
    workflow_1.save(str(tmpdir / "one"))
    new_dataset = workflow_1.transform(dataset)

    workflow_ops_2 = selector >> wf_ops.Rename(postfix="_2")
    workflow_2 = nvt.Workflow(workflow_ops_2)
    workflow_2.fit(new_dataset)
    workflow_2.save(str(tmpdir / "two"))

    repository = "repository_path/"
    version = 1
    kind = ""
    config = {
        "parameters": {
            "operator_names": {"string_value": json.dumps(["WorkflowOp_1", "WorkflowOp_2"])},
            "WorkflowOp_1": {
                "string_value": json.dumps(
                    {
                        "module_name": "nvtabular.inference.graph.ops.workflow",
                        "class_name": "WorkflowOp",
                        "workflow_path": str(tmpdir / "one"),
                    }
                )
            },
            "WorkflowOp_2": {
                "string_value": json.dumps(
                    {
                        "module_name": "nvtabular.inference.graph.ops.workflow",
                        "class_name": "WorkflowOp",
                        "workflow_path": str(tmpdir / "two"),
                    }
                )
            },
        }
    }

    runner = OperatorRunner(repository, version, kind, config)

    assert len(runner.operators) == 2

    for idx, loaded_workflow_op in enumerate(runner.operators):
        loaded_workflow = loaded_workflow_op.workflow
        assert isinstance(loaded_workflow_op, WorkflowOp)
        assert len(loaded_workflow.output_schema.column_names) > 0
        assert all(
            str(idx + 1) in column_name
            for column_name in loaded_workflow.output_schema.column_names
        )

    assert loaded_workflow.output_schema == workflow_2.output_schema
