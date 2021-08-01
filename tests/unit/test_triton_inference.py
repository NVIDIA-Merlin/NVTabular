import contextlib
import os
import signal
import subprocess
import time
from distutils.spawn import find_executable

import numpy as np
import pandas as pd
import pytest

import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.dispatch import HAS_GPU, _hash_series, _make_df
from nvtabular.ops.operator import Supports
from tests.conftest import assert_eq

triton = pytest.importorskip("nvtabular.inference.triton")
data_conversions = pytest.importorskip("nvtabular.inference.triton.data_conversions")

grpcclient = pytest.importorskip("tritonclient.grpc")
tritonclient = pytest.importorskip("tritonclient")

TRITON_SERVER_PATH = find_executable("tritonserver")


BACKEND = "python"
if os.path.exists("/opt/tritonserver/backends/nvtabular/libtriton_nvtabular.so"):
    BACKEND = "nvtabular"


@contextlib.contextmanager
def run_triton_server(modelpath):
    cmdline = [
        TRITON_SERVER_PATH,
        "--model-repository",
        modelpath,
        "--backend-config=tensorflow,version=2",
    ]
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


def _verify_workflow_on_tritonserver(
    tmpdir, workflow, df, model_name, output_model="tensorflow", model_info=None
):
    """tests that the nvtabular workflow produces the same results when run locally in the
    process, and when run in tritonserver"""
    # fit the workflow and test on the input
    dataset = nvt.Dataset(df)
    workflow.fit(dataset)

    local_df = workflow.transform(dataset).to_ddf().compute(scheduler="synchronous")
    triton.generate_nvtabular_model(
        workflow=workflow,
        name=model_name,
        output_path=tmpdir + f"/{model_name}",
        version=1,
        output_model=output_model,
        output_info=model_info,
        backend=BACKEND,
    )

    inputs = triton.convert_df_to_triton_input(df.columns, df)
    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs)

        for col in df.columns:
            features = response.as_numpy(col)
            triton_df = _make_df({col: features.reshape(features.shape[0])})
            assert_eq(triton_df, local_df[[col]])


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
def test_error_handling(tmpdir):
    df = _make_df({"x": np.arange(10), "y": np.arange(10)})

    def custom_transform(col):
        if len(col) == 2:
            raise ValueError("Lets cause some problems")
        return col

    features = ["x", "y"] >> ops.FillMissing() >> ops.Normalize() >> custom_transform
    workflow = nvt.Workflow(features)
    workflow.fit(nvt.Dataset(df))

    model_name = "test_error_handling"
    triton.generate_nvtabular_model(
        workflow, model_name, tmpdir + f"/{model_name}", backend=BACKEND
    )

    with run_triton_server(tmpdir) as client:
        inputs = triton.convert_df_to_triton_input(["x", "y"], df[:2])
        with pytest.raises(tritonclient.utils.InferenceServerException) as exception_info:
            client.infer(model_name, inputs)

        assert "ValueError: Lets cause some problems" in str(exception_info.value)


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_tritonserver_inference_string(tmpdir, output_model):
    df = _make_df({"user": ["aaaa", "bbbb", "cccc", "aaaa", "bbbb", "aaaa"]})
    features = ["user"] >> ops.Categorify()
    workflow = nvt.Workflow(features)

    if output_model == "pytorch":
        model_info = {"user": {"columns": ["user"], "dtype": "int64"}}
    else:
        model_info = None
    _verify_workflow_on_tritonserver(
        tmpdir, workflow, df, "test_inference_string", output_model, model_info
    )


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_large_strings(tmpdir, output_model):
    strings = ["a" * (2 ** exp) for exp in range(1, 17)]
    df = _make_df({"description": strings})
    features = ["description"] >> ops.Categorify()
    workflow = nvt.Workflow(features)

    if output_model == "pytorch":
        model_info = {"description": {"columns": ["description"], "dtype": "int64"}}
    else:
        model_info = None
    _verify_workflow_on_tritonserver(
        tmpdir, workflow, df, "test_large_string", output_model, model_info
    )


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_concatenate_dataframe(tmpdir, output_model):
    # we were seeing an issue in the rossmann workflow where we dropped certain columns,
    # https://github.com/NVIDIA/NVTabular/issues/961
    df = _make_df(
        {
            "cat": ["aaaa", "bbbb", "cccc", "aaaa", "bbbb", "aaaa"],
            "cont": [0.0, 1.0, 2.0, 3.0, 4.0, 5],
        }
    )
    # this bug only happened with a dataframe representation: force this by using a lambda
    cats = ["cat"] >> ops.LambdaOp(lambda col: _hash_series(col) % 1000)
    conts = ["cont"] >> ops.Normalize() >> ops.FillMissing() >> ops.LogOp()
    workflow = nvt.Workflow(cats + conts)

    if output_model == "pytorch":
        model_info = {
            "cat": {"columns": ["cat"], "dtype": "int32"},
            "cont": {"columns": ["cont"], "dtype": "float32"},
        }
    else:
        model_info = None
    _verify_workflow_on_tritonserver(
        tmpdir, workflow, df, "test_concatenate_dataframe", output_model, model_info
    )


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_numeric_dtypes(tmpdir, output_model):
    if output_model == "pytorch":
        model_info = dict()
    else:
        model_info = None

    dtypes = []
    for width in [8, 16, 32, 64]:
        dtype = f"int{width}"
        dtypes.append((dtype, np.iinfo(dtype)))
        if output_model == "pytorch":
            model_info[dtype] = {"columns": [dtype], "dtype": dtype}

        dtype = f"uint{width}"
        dtypes.append((dtype, np.iinfo(dtype)))
        if output_model == "pytorch":
            model_info[dtype] = {"columns": [dtype], "dtype": dtype}

    for width in [32, 64]:
        dtype = f"float{width}"
        dtypes.append((dtype, np.finfo(dtype)))
        if output_model == "pytorch":
            model_info[dtype] = {"columns": [dtype], "dtype": dtype}

    def check_dtypes(col):
        assert str(col.dtype) == col.name
        return col

    # simple transform to make sure we can round-trip the min/max values for each dtype,
    # through triton, with the 'transform' here just checking that the dtypes are correct
    df = _make_df(
        {dtype: np.array([limits.max, 0, limits.min], dtype=dtype) for dtype, limits in dtypes}
    )
    features = nvt.ColumnGroup(df.columns) >> check_dtypes
    workflow = nvt.Workflow(features)
    _verify_workflow_on_tritonserver(
        tmpdir, workflow, df, "test_numeric_dtypes", output_model, model_info
    )


def test_generate_triton_multihot(tmpdir):
    df = _make_df(
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

    # save workflow to triton / verify we see some expected output
    repo = os.path.join(tmpdir, "models")
    triton.generate_nvtabular_model(workflow, "model", repo)
    workflow = None

    assert os.path.exists(os.path.join(repo, "config.pbtxt"))

    workflow = nvt.Workflow.load(os.path.join(repo, "1", "workflow"))
    transformed = workflow.transform(nvt.Dataset(df)).to_ddf().compute()

    assert_eq(expected, transformed)


@pytest.mark.parametrize("engine", ["parquet"])
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_generate_triton_model(tmpdir, engine, output_model, df):
    tmpdir = "./tmp"
    conts = ["x", "y", "id"] >> ops.FillMissing() >> ops.Normalize()
    cats = ["name-cat", "name-string"] >> ops.Categorify(cat_cache="host")
    workflow = nvt.Workflow(conts + cats)
    workflow.fit(nvt.Dataset(df))
    expected = workflow.transform(nvt.Dataset(df)).to_ddf().compute()

    # save workflow to triton / verify we see some expected output
    if output_model == "pytorch":
        model_info = {
            "name-cat": {"columns": ["name-cat"], "dtype": "int64"},
            "name-string": {"columns": ["name-string"], "dtype": "int64"},
            "id": {"columns": ["id"], "dtype": "float32"},
            "x": {"columns": ["x"], "dtype": "float32"},
            "y": {"columns": ["y"], "dtype": "float32"},
        }
    else:
        model_info = None

    repo = os.path.join(tmpdir, "models")
    triton.generate_nvtabular_model(
        workflow=workflow,
        name="model",
        output_path=repo,
        version=1,
        output_model=output_model,
        output_info=model_info,
    )
    workflow = None

    assert os.path.exists(os.path.join(repo, "config.pbtxt"))

    workflow = nvt.Workflow.load(os.path.join(repo, "1", "workflow"))
    transformed = workflow.transform(nvt.Dataset(df)).to_ddf().compute()
    assert_eq(expected, transformed)


# lets test the data format conversion function on the full cartesian product
# of the Support flags
_SUPPORTS = list(Supports)
if not HAS_GPU:
    _SUPPORTS = [s for s in _SUPPORTS if "GPU" not in str(s)]


@pytest.mark.parametrize("_from", _SUPPORTS)
@pytest.mark.parametrize("_to", _SUPPORTS)
def test_convert_format(_from, _to):
    convert_format = data_conversions.convert_format

    # we want to test conversion from '_from' to '_to' but this requires us roundtripping
    # from a known format. I'm picking pd -> _from -> _to -> pandas somewhat arbitrarily
    df = pd.DataFrame(
        {"float": [0.0, 1.0, 2.0], "int": [10, 11, 12], "multihot": [[0, 1, 2, 3], [3, 4], [5]]}
    )

    if _from != Supports.GPU_DICT_ARRAY and _to != Supports.GPU_DICT_ARRAY:
        df["string"] = ["aa", "bb", "cc"]
        df["multihot_string"] = [["aaaa", "bb", "cc"], ["dd", "ee"], ["fffffff"]]

    start, kind = convert_format(df, Supports.CPU_DATAFRAME, _from)
    assert kind == _from
    mid, kind = convert_format(start, kind, _to)
    assert kind == _to
    final, kind = convert_format(mid, kind, Supports.CPU_DATAFRAME)
    assert kind == Supports.CPU_DATAFRAME
    assert_eq(df, final)
