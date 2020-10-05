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

import cudf
import fsspec
import pytest
from cudf.tests.utils import assert_eq

import nvtabular as nvt
from nvtabular import ops as ops
from tests.conftest import mycols_csv, mycols_pq

boto3 = pytest.importorskip("boto3")
s3fs = pytest.importorskip("s3fs")
moto = pytest.importorskip("moto")


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture(scope="function")
def s3(aws_credentials):
    with moto.mock_s3():
        yield boto3.client("s3", region_name="us-east-1")


@pytest.mark.parametrize("engine", ["parquet", "csv"])
def test_s3_dataset(s3, paths, engine, df):
    # create a mocked out bucket here
    bucket = "testbucket"
    s3.create_bucket(Bucket=bucket)

    s3_paths = []
    for path in paths:
        s3_path = f"s3://{bucket}/{path}"
        with fsspec.open(s3_path, "wb") as f:
            f.write(open(path, "rb").read())
        s3_paths.append(s3_path)

    # create a basic s3 dataset
    dataset = nvt.Dataset(s3_paths)

    # make sure the iteration API works
    columns = mycols_pq if engine == "parquet" else mycols_csv
    gdf = cudf.concat(list(dataset.to_iter()))[columns]
    assert_eq(gdf.reset_index(drop=True), df.reset_index(drop=True))

    cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
    cont_names = ["x", "y", "id"]
    label_name = ["label"]

    processor = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=label_name)

    processor.add_feature([ops.FillMissing(), ops.Clip(min_value=0), ops.LogOp()])
    processor.add_preprocess(ops.Normalize())
    processor.add_preprocess(ops.Categorify(cat_cache="host"))
    processor.finalize()

    processor.update_stats(dataset)
