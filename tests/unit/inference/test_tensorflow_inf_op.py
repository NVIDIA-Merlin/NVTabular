import os
import pathlib
from copy import deepcopy

import pytest

from nvtabular import ColumnSelector, Schema, graph

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

model_config = pytest.importorskip("nvtabular.inference.triton.model_config_pb2")
tf_op = pytest.importorskip("nvtabular.inference.graph.ops.tensorflow")

tf = pytest.importorskip("tensorflow")


def test_tf_op_exports_own_config(tmpdir):
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="input", dtype=tf.int32, shape=(784,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Triton
    triton_op = tf_op.TensorflowOp(model)
    triton_op.export(tmpdir, None)

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
        assert parsed.name == triton_op.name
        assert parsed.backend == "tensorflow"


def test_tf_op_compute_schema():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="input", dtype=tf.int32, shape=(784,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Triton
    triton_op = tf_op.TensorflowOp(model)

    out_schema = triton_op.compute_output_schema(Schema(["input"]), ColumnSelector(["input"]))
    assert out_schema.column_names == ["output"]


def test_tf_schema_validation():
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(name="input", dtype=tf.int32, shape=(784,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, name="output"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.SparseCategoricalAccuracy()],
    )

    # Triton
    tf_node = [] >> tf_op.TensorflowOp(model)
    tf_graph = graph.graph.Graph(tf_node)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(tf_graph).fit_schema(Schema(["input", "not_input"]))
    assert "Request schema provided to TensorflowOp" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(tf_graph).fit_schema(Schema(["not_input"]))
    assert "Request schema provided to TensorflowOp" in str(exception_info.value)

    with pytest.raises(ValueError) as exception_info:
        deepcopy(tf_graph).fit_schema(Schema([]))
    assert "Request schema provided to TensorflowOp" in str(exception_info.value)
