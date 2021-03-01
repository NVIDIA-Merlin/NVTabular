import os

import pytest
from cudf.tests.utils import assert_eq

import nvtabular as nvt
import nvtabular.ops as ops

#triton = pytest.importorskip("nvtabular.inference.triton")

@pytest.mark.skip(reason="not playing nice with others")
@pytest.mark.parametrize("engine", ["parquet"])
def test_generate_triton_model(tmpdir, engine, df):
    tmpdir = "./tmp"
    conts = ["x", "y", "id"] >> ops.FillMissing() >> ops.Normalize()
    cats = ["name-cat", "name-string"] >> ops.Categorify(cat_cache="host")
    workflow = nvt.Workflow(conts + cats)
    workflow.fit(nvt.Dataset(df))
    expected = workflow.transform(nvt.Dataset(df)).to_ddf().compute()

    # save workflow to triton / verify we see some expected output
    repo = os.path.join(tmpdir, "models")
    triton.generate_triton_model(workflow, "model", repo)
    workflow = None

    assert os.path.exists(os.path.join(repo, "config.pbtxt"))

    workflow = nvt.Workflow.load(os.path.join(repo, "1", "workflow"))
    transformed = workflow.transform(nvt.Dataset(df)).to_ddf().compute()

    assert_eq(expected, transformed)
