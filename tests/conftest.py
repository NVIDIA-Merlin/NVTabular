#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import contextlib
import glob
import os
import platform
import random
import signal
import socket
import subprocess
import time

import dask
import numpy as np
import pandas as pd

try:
    import cudf

    try:
        import cudf.testing._utils

        assert_eq = cudf.testing._utils.assert_eq
    except ImportError:
        import cudf.tests.utils

        assert_eq = cudf.tests.utils.assert_eq
except ImportError:
    cudf = None

    def assert_eq(a, b, *args, **kwargs):
        if isinstance(a, pd.DataFrame):
            return pd.testing.assert_frame_equal(a, b, *args, **kwargs)
        elif isinstance(a, pd.Series):
            return pd.testing.assert_series_equal(a, b, *args, **kwargs)
        else:
            return np.testing.assert_allclose(a, b)


import psutil
import pytest
from asvdb import ASVDb, BenchmarkInfo, utils
from dask.distributed import Client, LocalCluster
from numba import cuda

import nvtabular

allcols_csv = ["timestamp", "id", "label", "name-string", "x", "y", "z"]
mycols_csv = ["name-string", "id", "label", "x", "y"]
mycols_pq = ["name-cat", "name-string", "id", "label", "x", "y"]
mynames = [
    "Alice",
    "Bob",
    "Charlie",
    "Dan",
    "Edith",
    "Frank",
    "Gary",
    "Hannah",
    "Ingrid",
    "Jerry",
    "Kevin",
    "Laura",
    "Michael",
    "Norbert",
    "Oliver",
    "Patricia",
    "Quinn",
    "Ray",
    "Sarah",
    "Tim",
    "Ursula",
    "Victor",
    "Wendy",
    "Xavier",
    "Yvonne",
    "Zelda",
]

_CUDA_CLUSTER = None


@pytest.fixture(scope="module")
def client():
    cluster = LocalCluster(n_workers=2)
    client = Client(cluster)
    yield client
    client.close()
    cluster.close()


@contextlib.contextmanager
def get_cuda_cluster():
    from dask_cuda import LocalCUDACluster

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    n_workers = min(2, len(CUDA_VISIBLE_DEVICES.split(",")))
    cluster = LocalCUDACluster(n_workers=n_workers)
    yield cluster
    cluster.close()


@pytest.fixture(scope="session")
def datasets(tmpdir_factory):
    _lib = cudf if cudf else pd
    _datalib = cudf if cudf else dask
    df = _datalib.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-04",
        freq="60s",
        dtypes={
            "name-cat": str,
            "name-string": str,
            "id": int,
            "label": int,
            "x": float,
            "y": float,
            "z": float,
        },
    ).reset_index()

    if _datalib is dask:
        df = df.compute()

    df["name-string"] = _lib.Series(np.random.choice(mynames, df.shape[0])).astype("O")

    # Add two random null values to each column
    imax = len(df) - 1
    for col in df.columns:
        if col in ["name-cat", "label", "id"]:
            break
        for _ in range(2):
            rand_idx = random.randint(1, imax - 1)
            if rand_idx == df[col].shape[0] // 2:
                # dont want null in median
                rand_idx += 1
            df[col].iloc[rand_idx] = None

    datadir = tmpdir_factory.mktemp("data_test")
    datadir = {
        "parquet": tmpdir_factory.mktemp("parquet"),
        "csv": tmpdir_factory.mktemp("csv"),
        "csv-no-header": tmpdir_factory.mktemp("csv-no-header"),
        "cats": tmpdir_factory.mktemp("cats"),
    }

    half = int(len(df) // 2)

    # Write Parquet Dataset
    df.iloc[:half].to_parquet(str(datadir["parquet"].join("dataset-0.parquet")), chunk_size=1000)
    df.iloc[half:].to_parquet(str(datadir["parquet"].join("dataset-1.parquet")), chunk_size=1000)

    # Write CSV Dataset (Leave out categorical column)
    df.iloc[:half].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv"].join("dataset-0.csv")), index=False
    )
    df.iloc[half:].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv"].join("dataset-1.csv")), index=False
    )
    df.iloc[:half].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv-no-header"].join("dataset-0.csv")), header=False, index=False
    )
    df.iloc[half:].drop(columns=["name-cat"]).to_csv(
        str(datadir["csv-no-header"].join("dataset-1.csv")), header=False, index=False
    )

    return datadir


@pytest.fixture(scope="function")
def paths(engine, datasets):
    return sorted(glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0]))


@pytest.fixture(scope="function")
def df(engine, paths):
    _lib = cudf if cudf else pd
    if engine == "parquet":
        df1 = _lib.read_parquet(paths[0])[mycols_pq]
        df2 = _lib.read_parquet(paths[1])[mycols_pq]
    elif engine == "csv-no-header":
        df1 = _lib.read_csv(paths[0], header=None, names=allcols_csv)[mycols_csv]
        df2 = _lib.read_csv(paths[1], header=None, names=allcols_csv)[mycols_csv]
    elif engine == "csv":
        df1 = _lib.read_csv(paths[0], header=0)[mycols_csv]
        df2 = _lib.read_csv(paths[1], header=0)[mycols_csv]
    else:
        raise ValueError("unknown engine:" + engine)

    gdf = _lib.concat([df1, df2], axis=0)
    gdf["id"] = gdf["id"].astype("int64")
    return gdf


@pytest.fixture(scope="function")
def dataset(request, paths, engine):
    try:
        gpu_memory_frac = request.getfixturevalue("gpu_memory_frac")
    except Exception:  # pylint: disable=broad-except
        gpu_memory_frac = 0.01

    try:
        cpu = request.getfixturevalue("cpu")
    except Exception:  # pylint: disable=broad-except
        cpu = False

    kwargs = {}
    if engine == "csv-no-header":
        kwargs["names"] = allcols_csv

    return nvtabular.Dataset(paths, part_mem_fraction=gpu_memory_frac, cpu=cpu, **kwargs)


@pytest.fixture(scope="session")
def asv_db():
    # Create an interface to an ASV "database" to write the results to.
    (repo, branch) = utils.getRepoInfo()  # gets repo info from CWD by default
    # allows control of results location
    db_dir = os.environ.get("ASVDB_DIR", "./benchmarks")
    db = ASVDb(dbDir=db_dir, repo=repo, branches=[branch])

    return db


@pytest.fixture(scope="session")
def bench_info():

    # Create a BenchmarkInfo object describing the benchmarking environment.
    # This can/should be reused when adding multiple results from the same environment.

    uname = platform.uname()
    (commitHash, commitTime) = utils.getCommitInfo()  # gets commit info from CWD by default
    cuda_version = os.environ["CUDA_VERSION"]
    # get GPU info from nvidia-smi

    bInfo = BenchmarkInfo(
        machineName=socket.gethostname(),
        cudaVer=cuda_version,
        osType="%s %s" % (uname.system, uname.release),
        pythonVer=platform.python_version(),
        commitHash=commitHash,
        commitTime=commitTime,
        gpuType=cuda.get_current_device().name.decode("utf-8"),
        cpuType=uname.processor,
        arch=uname.machine,
        ram="%d" % psutil.virtual_memory().total,
    )
    return bInfo


def get_cats(workflow, col, stat_name="categories", cpu=False):
    _lib = cudf if cudf and not cpu else pd
    # figure out the categorify node from the workflow graph
    cats = [
        cg.op
        for cg in nvtabular.column_group.iter_nodes([workflow.column_group])
        if isinstance(cg.op, nvtabular.ops.Categorify)
    ]
    if len(cats) != 1:
        raise RuntimeError(f"Found {len(cats)} categorical ops, expected 1")
    filename = cats[0].categories[col]
    df = _lib.read_parquet(filename)
    df.reset_index(drop=True, inplace=True)
    if cudf and not cpu:
        return df[col].values_host
    else:
        return df[col]


@contextlib.contextmanager
def run_triton_server(
    modelpath, model_name, triton_server_path, device_id="0", backend="tensorflow", ps_path=None
):
    import tritonclient
    import tritonclient.grpc as grpcclient

    if backend == "tensorflow":
        backend_config = "tensorflow,version=2"
    elif backend == "hugectr":
        backend_config = "hugectr,ps=" + ps_path
    else:
        raise ValueError("unknown backend:" + backend)

    cmdline = [
        triton_server_path,
        "--model-repository",
        modelpath,
        "--backend-config",
        backend_config,
        "--model-control-mode=explicit",
        "--load-model",
        model_name,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_id
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


def run_in_context(func, *args, context=None, **kwargs):
    # Convenience utility to execute a function within
    # a specific `context`.  For example, this can be
    # used to test that a function raises a `UserWarning`
    # by setting `context=pytest.warns(UserWarning)`
    if context is None:
        context = contextlib.suppress()
    with context:
        result = func(*args, **kwargs)
    return result
