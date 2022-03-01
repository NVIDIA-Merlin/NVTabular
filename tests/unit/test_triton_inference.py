import contextlib
import os
import signal
import subprocess
import time
from distutils.spawn import find_executable

import numpy as np
import pandas as pd
import pytest
from merlin.dag import Supports

import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular import ColumnSelector
from nvtabular.dispatch import HAS_GPU, hash_series, make_df
from tests.conftest import assert_eq

triton = pytest.importorskip("nvtabular.inference.triton")
data_conversions = pytest.importorskip("nvtabular.inference.triton.data_conversions")
ensemble = pytest.importorskip("nvtabular.inference.triton.ensemble")

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
    tmpdir,
    workflow,
    df,
    model_name,
    output_model="tensorflow",
    sparse_max=None,
    cats=None,
    conts=None,
):
    """tests that the nvtabular workflow produces the same results when run locally in the
    process, and when run in tritonserver"""
    # fit the workflow and test on the input
    dataset = nvt.Dataset(df)

    if not workflow.output_dtypes:
        workflow.fit(dataset)

    local_df = workflow.transform(dataset).to_ddf().compute(scheduler="synchronous")

    triton.generate_nvtabular_model(
        workflow=workflow,
        name=model_name,
        output_path=tmpdir + f"/{model_name}",
        version=1,
        output_model=output_model,
        sparse_max=sparse_max,
        backend=BACKEND,
        cats=cats,
        conts=conts,
    )

    inputs = triton.convert_df_to_triton_input(df.columns, df)

    outputs = []
    for col_name, col_schema in workflow.output_schema.column_schemas.items():
        if col_schema.is_list and col_schema.is_ragged:
            outputs.append(f"{col_name}__values")
            outputs.append(f"{col_name}__nnzs")
        else:
            outputs.append(col_name)
    outputs = [grpcclient.InferRequestedOutput(col) for col in outputs]

    with run_triton_server(tmpdir) as client:
        response = client.infer(model_name, inputs, outputs=outputs)

        for col in workflow.output_dtypes.keys():
            features = response.as_numpy(col)
            if sparse_max and col in sparse_max:
                features = features.tolist()
                triton_df = make_df()
                triton_df[col] = features
            else:
                triton_df = make_df({col: features.reshape(features.shape[0])})
            assert_eq(triton_df, local_df[[col]])


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
def test_error_handling(tmpdir):
    df = make_df({"x": np.arange(10), "y": np.arange(10)})

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
    df = make_df({"user": ["aaaa", "bbbb", "cccc", "aaaa", "bbbb", "aaaa"]})
    features = ["user"] >> ops.Categorify()
    workflow = nvt.Workflow(features)

    _verify_workflow_on_tritonserver(
        tmpdir,
        workflow,
        df,
        "test_inference_string",
        output_model,
    )


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_large_strings(tmpdir, output_model):
    strings = ["a" * (2 ** exp) for exp in range(1, 17)]
    df = make_df({"description": strings})
    features = ["description"] >> ops.Categorify()
    workflow = nvt.Workflow(features)
    workflow.fit(nvt.Dataset(df))

    _verify_workflow_on_tritonserver(
        tmpdir,
        workflow,
        df,
        "test_large_string",
        output_model,
    )


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def testconcatenate_dataframe(tmpdir, output_model):
    # we were seeing an issue in the rossmann workflow where we dropped certain columns,
    # https://github.com/NVIDIA/NVTabular/issues/961
    df = make_df(
        {
            "cat": ["aaaa", "bbbb", "cccc", "aaaa", "bbbb", "aaaa"],
            "cont": [0.0, 1.0, 2.0, 3.0, 4.0, 5],
        }
    )
    # this bug only happened with a dataframe representation: force this by using a lambda
    cats = ["cat"] >> ops.LambdaOp(lambda col: hash_series(col) % 1000)
    conts = ["cont"] >> ops.Normalize() >> ops.FillMissing() >> ops.LogOp()

    workflow = nvt.Workflow(cats + conts)

    _verify_workflow_on_tritonserver(
        tmpdir,
        workflow,
        df,
        "test_concatenate_dataframe",
        output_model,
        cats=["cat"],
        conts=["cont"],
    )


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_numeric_dtypes(tmpdir, output_model):
    def make_dtypes(prefix, widths, info_type):
        return [(f"{prefix}{width}", info_type(f"{prefix}{width}")) for width in widths]

    int_dtypes = make_dtypes("int", [32, 64], np.iinfo)
    uint_dtypes = make_dtypes("uint", [32, 64], np.iinfo)
    float_dtypes = make_dtypes("float", [32, 64], np.finfo)

    def check_dtypes(col):
        assert str(col.dtype) == col.name
        return col

    # simple transform to make sure we can round-trip the min/max values for each dtype,
    # through triton, with the 'transform' here just checking that the dtypes are correct
    dtypes = int_dtypes + uint_dtypes + float_dtypes
    df = make_df(
        {dtype: np.array([limits.max, 0, limits.min], dtype=dtype) for dtype, limits in dtypes}
    )

    features = nvt.ColumnSelector(df.columns) >> ops.LambdaOp(check_dtypes)
    workflow = nvt.Workflow(features)
    dataset = nvt.Dataset(df)
    workflow.fit(dataset)

    if output_model == "pytorch":
        for dtype, _ in dtypes:
            workflow.output_dtypes[dtype] = np.dtype(dtype)

    _verify_workflow_on_tritonserver(
        tmpdir,
        workflow,
        df,
        "test_numeric_dtypes",
        output_model,
        cats=[dtype for dtype, _ in int_dtypes + uint_dtypes],
        conts=[dtype for dtype, _ in float_dtypes],
    )


def test_generate_triton_multihot(tmpdir):
    df = make_df(
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

    repo = os.path.join(tmpdir, "models")
    triton.generate_nvtabular_model(
        workflow=workflow,
        name="model",
        output_path=repo,
        version=1,
        output_model=output_model,
    )
    workflow = None

    assert os.path.exists(os.path.join(repo, "config.pbtxt"))

    workflow = nvt.Workflow.load(os.path.join(repo, "1", "workflow"))
    transformed = workflow.transform(nvt.Dataset(df)).to_ddf().compute()
    assert_eq(expected, transformed)


def test_remove_columns():
    # _remove_columns was failing to export the criteo example, because
    # the label column was getting inserted into the subgroups of the output node
    # https://github.com/NVIDIA-Merlin/NVTabular/issues/1198
    label_columns = ["label"]
    cats = ["a"] >> ops.Categorify()
    conts = ["b"] >> ops.Normalize()
    workflow = nvt.Workflow(cats + conts + label_columns)

    df = pd.DataFrame({"a": ["a", "b"], "b": [1.0, 2.0], "label": [0, 1]})
    workflow.fit(nvt.Dataset(df))

    removed = workflow.remove_inputs(label_columns)
    assert set(removed.output_dtypes.keys()) == {"a", "b"}


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


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow", "pytorch"])
def test_groupby_model(tmpdir, output_model):
    size = 20
    df = make_df(
        {
            "id": np.random.choice([0, 1], size=size),
            "ts": np.linspace(0.0, 10.0, num=size),
            "x": np.arange(size),
            "y": np.linspace(0.0, 10.0, num=size),
        }
    )

    groupby_features = ColumnSelector(["id", "ts", "x", "y"]) >> ops.Groupby(
        groupby_cols=["id"],
        sort_cols=["ts"],
        aggs={
            "x": ["sum"],
            "y": ["first"],
        },
        name_sep="-",
    )
    workflow = nvt.Workflow(groupby_features)

    _verify_workflow_on_tritonserver(
        tmpdir, workflow, df, "groupby", output_model, cats=["id", "y-first"], conts=["x-sum"]
    )


@pytest.mark.skipif(TRITON_SERVER_PATH is None, reason="Requires tritonserver on the path")
@pytest.mark.parametrize("output_model", ["tensorflow"])
def test_seq_etl_tf_model(tmpdir, output_model):
    size = 100
    max_length = 10
    df = make_df(
        {
            "id": np.random.choice([0, 1], size=size),
            "item_id": np.random.randint(1, 10, size),
            "ts": np.linspace(0.0, 10.0, num=size).astype(np.float32),
            "y": np.linspace(0.0, 10.0, num=size).astype(np.float32),
        }
    )

    groupby_features = ColumnSelector(["id", "item_id", "ts", "y"]) >> ops.Groupby(
        groupby_cols=["id"],
        sort_cols=["ts"],
        aggs={
            "item_id": ["list"],
            "y": ["list"],
        },
        name_sep="-",
    )
    feats_list = groupby_features["item_id-list", "y-list"]
    feats_trim = feats_list >> ops.ListSlice(0, max_length, pad=True)
    selected_features = groupby_features["id"] + feats_trim

    workflow = nvt.Workflow(selected_features)

    sparse_max = {"item_id-list": max_length, "y-list": max_length}

    _verify_workflow_on_tritonserver(
        tmpdir,
        workflow,
        df,
        "groupby",
        output_model,
        sparse_max,
        cats=["id", "item_id-list"],
        conts=["y-list"],
    )
