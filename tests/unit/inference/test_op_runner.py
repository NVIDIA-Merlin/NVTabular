import json
import os

import numpy as np
import pytest

import nvtabular as nvt
import nvtabular.ops as wf_ops
from nvtabular.graph.graph import Graph
from tests.unit.inference.inference_utils import PlusTwoOp

op_runner = pytest.importorskip("nvtabular.inference.graph.op_runner")
inf_op = pytest.importorskip("nvtabular.inference.graph.ops.operator")


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
            "operator_names": {"string_value": json.dumps(["PlusTwoOp_1"])},
            "PlusTwoOp_1": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
        }
    }

    runner = op_runner.OperatorRunner(repository, version, kind, config)

    loaded_op = runner.operators[0]
    assert isinstance(loaded_op, PlusTwoOp)


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_loads_multiple_ops_same(tmpdir, dataset, engine):
    # NVT
    schema = dataset.schema
    for name in schema.column_names:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [nvt.graph.tags.Tags.USER]
        )

    repository = "repository_path/"
    version = 1
    kind = ""
    config = {
        "parameters": {
            "operator_names": {"string_value": json.dumps(["PlusTwoOp_1", "PlusTwoOp_2"])},
            "PlusTwoOp_1": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
            "PlusTwoOp_2": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
        }
    }

    runner = op_runner.OperatorRunner(config, repository, version, kind)

    assert len(runner.operators) == 2

    for idx, loaded_op in enumerate(runner.operators):
        assert isinstance(loaded_op, PlusTwoOp)


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_loads_multiple_ops_same_execute(tmpdir, dataset, engine):
    # NVT
    schema = dataset.schema
    for name in schema.column_names:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [nvt.graph.tags.Tags.USER]
        )

    repository = "repository_path/"
    version = 1
    kind = ""
    config = {
        "parameters": {
            "operator_names": {"string_value": json.dumps(["PlusTwoOp_1", "PlusTwoOp_2"])},
            "PlusTwoOp_1": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
            "PlusTwoOp_2": {
                "string_value": json.dumps(
                    {
                        "module_name": PlusTwoOp.__module__,
                        "class_name": "PlusTwoOp",
                    }
                )
            },
        }
    }

    runner = op_runner.OperatorRunner(config, repository, version, kind)

    inputs = {}
    for col_name in schema.column_names:
        inputs[col_name] = np.random.randint(10)

    outputs = runner.execute(inf_op.InferenceDataFrame(inputs))

    assert outputs["x+2+2"] == inputs["x"] + 4


@pytest.mark.parametrize("engine", ["parquet"])
def test_op_runner_single_node_export(tmpdir, dataset, engine):
    # assert against produced config
    schema = dataset.schema
    for name in schema.column_names:
        dataset.schema.column_schemas[name] = dataset.schema.column_schemas[name].with_tags(
            [nvt.graph.tags.Tags.USER]
        )

    inputs = ["x", "y"]

    node = inputs >> PlusTwoOp()

    graph = Graph(node)
    graph.fit_schema(dataset.schema)

    config = node.export(tmpdir)

    file_path = os.path.join(str(tmpdir), "config.pbtxt")

    assert os.path.exists(file_path)
    config_file = open(file_path, "r").read()
    assert config_file == str(config)
    assert len(config.input) == len(inputs)
    assert len(config.output) == len(inputs)
    for idx, conf in enumerate(config.output):
        assert conf.name == inputs[idx] + "+2"
    breakpoint()
