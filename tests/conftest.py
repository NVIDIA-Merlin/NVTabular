#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import random

import cudf
import numpy as np
import pytest
from dask.distributed import Client, LocalCluster

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
    client = Client(LocalCluster(n_workers=2))
    yield client
    client.close()


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
    df = cudf.datasets.timeseries(
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
    df["name-string"] = cudf.Series(np.random.choice(mynames, df.shape[0])).astype("O")

    # Add two random null values to each column
    imax = len(df) - 1
    for col in df.columns:
        if col in ["name-cat", "label", "id"]:
            break
        df[col].iloc[random.randint(1, imax - 1)] = None
        df[col].iloc[random.randint(1, imax - 1)] = None

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
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
        df2 = cudf.read_parquet(paths[1])[mycols_pq]
    elif engine == "csv-no-header":
        df1 = cudf.read_csv(paths[0], header=None, names=allcols_csv)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=None, names=allcols_csv)[mycols_csv]
    elif engine == "csv":
        df1 = cudf.read_csv(paths[0], header=0)[mycols_csv]
        df2 = cudf.read_csv(paths[1], header=0)[mycols_csv]
    else:
        raise ValueError("unknown engine:" + engine)

    gdf = cudf.concat([df1, df2], axis=0)
    gdf["id"] = gdf["id"].astype("int64")
    return gdf


@pytest.fixture(scope="function")
def dataset(request, paths, engine):
    try:
        gpu_memory_frac = request.getfixturevalue("gpu_memory_frac")
    except Exception:
        gpu_memory_frac = 0.01

    kwargs = {}
    if engine == "csv-no-header":
        kwargs["names"] = allcols_csv

    return nvtabular.Dataset(paths, part_mem_fraction=gpu_memory_frac, **kwargs)


def get_cats(processor, col, stat_name="categories"):
    if isinstance(processor, nvtabular.workflow.Workflow):
        filename = processor.stats[stat_name][col]
        gdf = cudf.read_parquet(filename)
        gdf.reset_index(drop=True, inplace=True)
        return gdf[col].values_host
    else:
        return processor.stats["encoders"][col].get_cats().values_host
