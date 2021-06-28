import contextlib
import os
import signal
import subprocess
import time
from distutils.spawn import find_executable

import cudf
import numpy as np
import pytest
from cudf.tests.utils import assert_eq

import nvtabular as nvt
import nvtabular.ops as ops

triton = pytest.importorskip("nvtabular.inference.triton")
grpcclient = pytest.importorskip("tritonclient.grpc")
tritonclient = pytest.importorskip("tritonclient")

TRITON_SERVER_PATH = find_executable("tritonserver")


BACKEND = "python"
if os.path.exists("/opt/tritonserver/backends/nvtabular/libtriton_nvtabular.so"):
    BACKEND = "nvtabular"


@contextlib.contextmanager
def run_triton_server(modelpath):
    cmdline = [TRITON_SERVER_PATH, "--model-repository", modelpath]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with subprocess.Popen(cmdline, env=env) as process:
        try:
            with grpcclient.InferenceServerClient("localhost:8001") as client:
                # wait until server is ready
                for _ in range(60):
                    if process.poll() is not None:
                        retcode = process.returncode
                        raise RuntimeError(f"Tritonserver failed to start (ret={retcode})")

                    try:
                        ready = client.is_server_ready()
                    except tritonclient.utils.InferenceServerException:
                        ready = False

                    if ready:
                        yield client
                        return

                    time.sleep(1)

                raise RuntimeError("Timed out waiting for tritonserver to become ready")
        finally:
            # signal triton to shutdown
            process.send_signal(signal.SIGINT)


def _verify_workflow_on_tritonserver(tmpdir, workflow, df, model_name):
    """tests that the nvtabular workflow produces the same results when run locally in the
    process, and when run in tritonserver"""
    # fit the workflow and test on the input
    dataset = nvt.Dataset(df)
    workflow.fit(dataset)

    local_df = workflow.transform(dataset).to_ddf().compute(scheduler="synchronous")
    triton.generate_nvtabular_model(
        workflow, model_name, tmpdir + f"/{model_name}", backend=BACKEND
    )

    inputs = triton.convert_df_to_triton_input(df.columns, df)
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs)

        for col in df.columns:
            features = response.as_numpy(col)
            triton_df = cudf.DataFrame({col: features.reshape(features.shape[0])})
            assert_eq(triton_df, local_df[[col]])


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
def test_error_handling(tmpdir):
    df = cudf.DataFrame({"x": np.arange(10), "y": np.arange(10)})

    def custom_transform(col):
        if len(col) == 2:
            raise ValueError("Lets cause some problems")
        return col

    features = ["x", "y"] >> ops.FillMissing() >> ops.Normalize() >> custom_transform
    workflow = nvt.Workflow(features)
    workflow.fit(nvt.Dataset(df))

    model_name = "test_error_handling"
    triton.generate_nvtabular_model(
        workflow, model_name, tmpdir + f"/{model_name}", backend="nvtabular"
    )

    with run_triton_server(tmpdir) as client:
        inputs = triton.convert_df_to_triton_input(["x", "y"], df[:2])
        with pytest.raises(tritonclient.utils.InferenceServerException) as exception_info:
            client.infer(model_name, inputs)

        assert "ValueError: Lets cause some problems" in str(exception_info.value)


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
def test_tritonserver_inference_string(tmpdir):
    df = cudf.DataFrame({"user": ["aaaa", "bbbb", "cccc", "aaaa", "bbbb", "aaaa"]})
    features = ["user"] >> ops.Categorify()
    workflow = nvt.Workflow(features)
    _verify_workflow_on_tritonserver(tmpdir, workflow, df, "test_inference_string")


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
def test_large_strings(tmpdir):
    strings = ["a" * (2 ** exp) for exp in range(1, 17)]
    df = cudf.DataFrame({"description": strings})
    features = ["description"] >> ops.Categorify()
    workflow = nvt.Workflow(features)
    _verify_workflow_on_tritonserver(tmpdir, workflow, df, "test_large_string")


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
def test_numeric_dtypes(tmpdir):
    dtypes = []
    for width in [8, 16, 32, 64]:
        dtype = f"int{width}"
        dtypes.append((dtype, np.iinfo(dtype)))
        dtype = f"uint{width}"
        dtypes.append((dtype, np.iinfo(dtype)))

    for width in [32, 64]:
        dtype = f"float{width}"
        dtypes.append((dtype, np.finfo(dtype)))

    def check_dtypes(col):
        assert str(col.dtype) == col.name
        return col

    # simple transform to make sure we can round-trip the min/max values for each dtype,
    # through triton, with the 'transform' here just checking that the dtypes are correct
    df = cudf.DataFrame(
        {dtype: np.array([limits.max, 0, limits.min], dtype=dtype) for dtype, limits in dtypes}
    )
    features = nvt.ColumnGroup(df.columns) >> check_dtypes
    workflow = nvt.Workflow(features)
    _verify_workflow_on_tritonserver(tmpdir, workflow, df, "test_numeric_dtypes")


def test_generate_triton_multihot(tmpdir):
    df = cudf.DataFrame(
        {
            "userId": ["a", "a", "b"],
            "movieId": ["1", "2", "2"],
            "genres": [["action", "adventure"], ["action", "comedy"], ["comedy"]],
        }
    )

    cats = ["userId", "movieId", "genres"] >> nvt.ops.Categorify()
    workflow = nvt.Workflow(cats)
    workflow.fit(nvt.Dataset(df))
    expected = workflow.transform(nvt.Dataset(df)).to_ddf().compute()
    print(expected)

    # save workflow to triton / verify we see some expected output
    repo = os.path.join(tmpdir, "models")
    triton.generate_nvtabular_model(workflow, "model", repo)
    workflow = None

    assert os.path.exists(os.path.join(repo, "config.pbtxt"))

    workflow = nvt.Workflow.load(os.path.join(repo, "1", "workflow"))
    transformed = workflow.transform(nvt.Dataset(df)).to_ddf().compute()

    assert_eq(expected, transformed)


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
    triton.generate_nvtabular_model(workflow, "model", repo)
    workflow = None

    assert os.path.exists(os.path.join(repo, "config.pbtxt"))

    workflow = nvt.Workflow.load(os.path.join(repo, "1", "workflow"))
    transformed = workflow.transform(nvt.Dataset(df)).to_ddf().compute()

    assert_eq(expected, transformed)
