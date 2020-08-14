import cudf
import cupy
import pytest

from nvtabular.column_similarity import ColumnSimilarity


@pytest.mark.parametrize("on_device", [True, False])
@pytest.mark.parametrize("metric", ["tfidf", "cosine", "inner"])
def test_column_similarity(on_device, metric):
    categories = cupy.sparse.coo_matrix(
        (
            cupy.ones(10),
            (
                cupy.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3]),
                cupy.array([0, 1, 2, 1, 2, 3, 3, 4, 5, 1]),
            ),
        )
    )

    op = ColumnSimilarity("output", "left", categories, "right", metric=metric, on_device=on_device)
    df = op.apply_op(cudf.DataFrame({"left": [0, 0, 0, 0], "right": [0, 1, 2, 3]}), None, None)

    output = df.output.values
    if metric in ("tfidf", "cosine"):
        # distance from document 0 to itself should be 1, since these metrics are fully normalized
        assert float(output[0]) == pytest.approx(1)

    # distance from document 0 to document 2 should be 0 since they have no features in common
    assert output[2] == 0
