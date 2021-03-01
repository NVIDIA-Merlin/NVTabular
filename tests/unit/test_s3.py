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

import os
import shlex
import subprocess
import time
from contextlib import contextmanager
from io import BytesIO

import cudf
import pytest
from cudf.tests.utils import assert_eq
from dask.dataframe.io.parquet.core import create_metadata_file

import nvtabular as nvt
from nvtabular import ops as ops
from tests.conftest import mycols_csv, mycols_pq

boto3 = pytest.importorskip("boto3")
s3fs = pytest.importorskip("s3fs")
moto = pytest.importorskip("moto")
requests = pytest.importorskip("requests")


@contextmanager
def ensure_safe_environment_variables():
    """
    Get a context manager to safely set environment variables
    All changes will be undone on close, hence environment variables set
    within this contextmanager will neither persist nor change global state.

    This function was copied from https://github.com/rapidsai/cudf
    (see cudf/python/dask_cudf/dask_cudf/io/tests/test_s3.py)
    """
    saved_environ = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


@pytest.fixture(scope="session")
def s3_base(worker_id):
    """
    Fixture to set up moto server in separate process

    This fixture was copied from https://github.com/rapidsai/cudf
    (see cudf/python/dask_cudf/dask_cudf/io/tests/test_s3.py)
    """
    with ensure_safe_environment_variables():
        # Fake aws credentials exported to prevent botocore looking for
        # system aws credentials, https://github.com/spulec/moto/issues/1793
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "foobar_key")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "foobar_secret")

        # Launching moto in server mode, i.e., as a separate process
        # with an S3 endpoint on localhost

        endpoint_port = 5000 if worker_id == "master" else 5550 + int(worker_id.lstrip("gw"))
        endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

        proc = subprocess.Popen(
            shlex.split(f"moto_server s3 -p {endpoint_port}"),
        )

        timeout = 5
        while timeout > 0:
            try:
                # OK to go once server is accepting connections
                r = requests.get(endpoint_uri)
                if r.ok:
                    break
            except Exception:
                pass
            timeout -= 0.1
            time.sleep(0.1)
        yield endpoint_uri

        proc.terminate()
        proc.wait()


@pytest.fixture()
def s3so(worker_id):
    """
    Returns s3 storage options to pass to fsspec

    This fixture was copied from https://github.com/rapidsai/cudf
    (see cudf/python/dask_cudf/dask_cudf/io/tests/test_s3.py)
    """
    endpoint_port = 5000 if worker_id == "master" else 5550 + int(worker_id.lstrip("gw"))
    endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

    return {"client_kwargs": {"endpoint_url": endpoint_uri}}


@contextmanager
def s3_context(s3_base, bucket, files=None):
    """
    This function was copied from https://github.com/rapidsai/cudf
    (see cudf/python/dask_cudf/dask_cudf/io/tests/test_s3.py)
    """
    if files is None:
        files = {}
    with ensure_safe_environment_variables():
        client = boto3.client("s3", endpoint_url=s3_base)
        client.create_bucket(Bucket=bucket, ACL="public-read-write")
        for f, data in files.items():
            client.put_object(Bucket=bucket, Key=f, Body=data)

        yield s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_base})

        for f, data in files.items():
            try:
                client.delete_object(Bucket=bucket, Key=f)
            except Exception:
                pass


@pytest.mark.parametrize("engine", ["parquet", "csv"])
def test_s3_dataset(s3_base, s3so, paths, datasets, engine, df):

    # Copy files to mock s3 bucket
    files = {}
    for i, path in enumerate(paths):
        with open(path, "rb") as f:
            fbytes = f.read()
        fn = path.split(os.path.sep)[-1]
        files[fn] = BytesIO()
        files[fn].write(fbytes)
        files[fn].seek(0)

    if engine == "parquet":
        # Workaround for nvt#539. In order to avoid the
        # bug in Dask's `create_metadata_file`, we need
        # to manually generate a "_metadata" file here.
        # This can be removed after dask#7295 is merged
        # (see https://github.com/dask/dask/pull/7295)
        fn = "_metadata"
        files[fn] = BytesIO()
        meta = create_metadata_file(
            paths,
            engine="pyarrow",
            out_dir=False,
        )
        meta.write_metadata_file(files[fn])
        files[fn].seek(0)

    with s3_context(s3_base=s3_base, bucket=engine, files=files):

        # Create nvt.Dataset from mock s3 paths
        url = f"s3://{engine}" if engine == "parquet" else f"s3://{engine}/*"
        dataset = nvt.Dataset(url, engine=engine, storage_options=s3so)

        # Check that the iteration API works
        columns = mycols_pq if engine == "parquet" else mycols_csv
        gdf = cudf.concat(list(dataset.to_iter()))[columns]
        assert_eq(gdf.reset_index(drop=True), df.reset_index(drop=True))

        cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
        cont_names = ["x", "y", "id"]
        label_name = ["label"]

        conts = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()
        cats = cat_names >> ops.Categorify(cat_cache="host")

        processor = nvt.Workflow(conts + cats + label_name)
        processor.fit(dataset)
