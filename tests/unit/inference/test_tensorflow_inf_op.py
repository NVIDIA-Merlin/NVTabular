import os
import pathlib

import pytest

from nvtabular import ColumnSelector, Schema

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa
from nvtabular.inference.graph.ops.tensorflow import TensorflowOp  # noqa

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
    triton_op = TensorflowOp(model)
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
        assert parsed.name == "tensorflow"
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
    triton_op = TensorflowOp(model)

    out_schema = triton_op.compute_output_schema(Schema(["input"]), ColumnSelector())
    assert out_schema.column_names == ["output"]
