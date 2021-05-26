import contextlib
import os
import signal
import subprocess
import time
from distutils.spawn import find_executable

import cudf
import pytest
from cudf.tests.utils import assert_eq

import nvtabular as nvt
import nvtabular.ops as ops

triton = pytest.importorskip("nvtabular.inference.triton")
grpcclient = pytest.importorskip("tritonclient.grpc")
tritonclient = pytest.importorskip("tritonclient")


_TRITON_SERVER_PATH = find_executable("tritonserver")


@contextlib.contextmanager
def run_triton_server(modelpath):
    cmdline = [_TRITON_SERVER_PATH, "--model-repository", modelpath]
    with subprocess.Popen(cmdline) as process:
        try:
            with grpcclient.InferenceServerClient("localhost:8001") as client:
                # wait until server is ready
                for _ in range(60):
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


@pytest.mark.skipif(_TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
def test_tritonserver_inference_string(tmpdir):
    df = cudf.DataFrame({"user": ["aaaa", "bbbb", "cccc", "aaaa", "bbbb", "aaaa"]})
    features = ["user"] >> ops.Categorify()
    workflow = nvt.Workflow(features)

    # fit the workflow and test on the input
    dataset = nvt.Dataset(df)
    workflow.fit(dataset)

    local_df = workflow.transform(dataset).to_ddf().compute(scheduler="synchronous")
    model_name = "test_inference_string"
    triton.generate_nvtabular_model(workflow, model_name, tmpdir + "/test_inference_string")

    inputs = triton.convert_df_to_triton_input(["user"], df)
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs)
        user_features = response.as_numpy("user")
        triton_df = cudf.DataFrame({"user": user_features.reshape(user_features.shape[0])})
        assert_eq(triton_df, local_df)


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
