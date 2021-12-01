import pytest

from nvtabular.graph.schema import Schema

tf = pytest.importorskip("tensorflow")
nvtf = pytest.importorskip("nvtabular.framework_utils.tensorflow")


def test_feature_column_utils():
    cols = [
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "vocab_1", ["a", "b", "c", "d"]
            ),
            16,
        ),
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                "vocab_2", ["1", "2", "3", "4", "5"]
            ),
            32,
        ),
    ]

    workflow, _ = nvtf.make_feature_column_workflow(cols, "target")

    schema = Schema(["vocab_1", "vocab_2", "target"])
    workflow.fit_schema(schema)
    assert workflow.output_node.output_schema.column_names == ["vocab_1", "vocab_2", "target"]
