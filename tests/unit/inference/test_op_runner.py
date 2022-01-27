import json

import numpy as np
import pytest

import nvtabular as nvt
import nvtabular.ops as wf_ops

op_runner = pytest.importorskip("nvtabular.inference.graph.op_runner")
inf_op = pytest.importorskip("nvtabular.inference.graph.ops.operator")


class PlusTwoOp(inf_op.InferenceOperator):
    @property
    def export_name(self):
        return str(self.__class__.__name__)

    def export(self, path):
        pass

    def transform(self, df: inf_op.InferenceDataFrame) -> inf_op.InferenceDataFrame:
        focus_df = df
        new_df = inf_op.InferenceDataFrame()
        for name, data in focus_df:
            new_df.tensors[f"{name}+2"] = data + 2
        return new_df

    @classmethod
    def from_config(cls, config):
        return PlusTwoOp()


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

    runner = op_runner.OperatorRunner(repository, version, kind, config)

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

    runner = op_runner.OperatorRunner(repository, version, kind, config)

    inputs = {}
    for col_name in schema.column_names:
        inputs[col_name] = np.random.randint(10)

    outputs = runner.execute(inf_op.InferenceDataFrame(inputs))

    assert outputs["x+2+2"] == inputs["x"] + 4
