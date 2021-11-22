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

import os
from io import BytesIO

import pytest
from dask.dataframe.io.parquet.core import create_metadata_file

import nvtabular as nvt
from nvtabular import ops
from tests.conftest import assert_eq, mycols_csv, mycols_pq

dask_cudf = pytest.importorskip("dask_cudf")
from dask_cudf.io.tests import test_s3  # noqa: E402

# Import fixtures and context managers from dask_cudf
s3_base = test_s3.s3_base
s3_context = test_s3.s3_context
s3so = test_s3.s3so


aiobotocore = pytest.importorskip("aiobotocore")


@pytest.fixture()
def patch_aiobotocore(s3so):
    """fixture to patch the aiobotocore client with the mock s3 endpoint url.
    Note that we should probably be passing the 'storage_options' along in the
    to_parquet call in nvt.Dataset - but as it stands now this is necessary to fully
    mock s3 objects"""

    # note we can't use moto 'mock_s3' here because of
    # https://github.com/aio-libs/aiobotocore/issues/755
    # but potentially when that's fixed we can drop this entire method

    def patched_create_client(*args, **kwargs):
        if "endpoint_url" not in kwargs:
            kwargs["endpoint_url"] = s3so["client_kwargs"]["endpoint_url"]
        return create_client(*args, **kwargs)

    create_client = aiobotocore.AioSession.create_client
    aiobotocore.AioSession.create_client = patched_create_client
    yield
    aiobotocore.AioSession.create_client = create_client


@pytest.mark.parametrize("engine", ["parquet", "csv"])
def test_s3_dataset(s3_base, s3so, paths, datasets, engine, df, patch_aiobotocore):
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

    with s3_context(s3_base=s3_base, bucket=engine, files=files) as s3fs:
        # Create nvt.Dataset from mock s3 paths
        url = f"s3://{engine}" if engine == "parquet" else f"s3://{engine}/*"
        dataset = nvt.Dataset(url, engine=engine, storage_options=s3so)

        # Check that the iteration API works
        columns = mycols_pq if engine == "parquet" else mycols_csv
        gdf = nvt.dispatch._concat(list(dataset.to_iter()))[columns]
        assert_eq(gdf.reset_index(drop=True), df.reset_index(drop=True))

        cat_names = ["name-cat", "name-string"] if engine == "parquet" else ["name-string"]
        cont_names = ["x", "y", "id"]
        label_name = ["label"]

        conts = cont_names >> ops.FillMissing() >> ops.Clip(min_value=0) >> ops.LogOp()
        cats = cat_names >> ops.Categorify(cat_cache="host")

        processor = nvt.Workflow(conts + cats + label_name)
        processor.fit(dataset)

        # make sure we can write out the dataset back to S3
        # (https://github.com/NVIDIA-Merlin/NVTabular/issues/1214)
        processor.transform(dataset).to_parquet(f"s3://{engine}/output")
        expected = processor.transform(dataset).to_ddf().compute()

        # make sure we can write out the workflow to s3
        processor.save(f"s3://{engine}/saved_workflow/")

        # make sure the workflow got saved to the right spot in S3
        workflow_files = s3fs.ls(f"/{engine}/saved_workflow/")
        assert workflow_files

        # finally make sure we can read in the workflow from S3, and use it
        # to transform values and get the same result as on the local fs
        reloaded = nvt.Workflow.load(f"s3://{engine}/saved_workflow/")
        from_s3 = reloaded.transform(dataset).to_ddf().compute()
        assert_eq(expected, from_s3)
