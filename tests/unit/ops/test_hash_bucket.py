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
import numpy as np
import pytest

import nvtabular as nvt
from nvtabular import dispatch, ops
from nvtabular.dispatch import HAS_GPU

if HAS_GPU:
    _CPU = [True, False]
    _HAS_GPU = True
else:
    _CPU = [True]
    _HAS_GPU = False


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1] if _HAS_GPU else [None])
@pytest.mark.parametrize("engine", ["parquet", "csv", "csv-no-header"])
@pytest.mark.parametrize("op_columns", [["name-string"], None])
@pytest.mark.parametrize("cpu", _CPU)
def test_hash_bucket(tmpdir, df, dataset, gpu_memory_frac, engine, op_columns, cpu):
    cat_names = ["name-string"]
    if cpu:
        dataset.to_cpu()
    if op_columns is None:
        num_buckets = 10
    else:
        num_buckets = {column: 10 for column in op_columns}

    hash_features = cat_names >> ops.HashBucket(num_buckets)
    processor = nvt.Workflow(hash_features)
    processor.fit(dataset)
    new_df = processor.transform(dataset).to_ddf().compute()

    # check sums for determinancy
    assert np.all(new_df[cat_names].values >= 0)
    assert np.all(new_df[cat_names].values <= 9)
    checksum = new_df[cat_names].sum().values

    new_df = processor.transform(dataset).to_ddf().compute()
    np.all(new_df[cat_names].sum().values == checksum)


@pytest.mark.skipif(not _HAS_GPU, reason="HashBucket doesn't work on lists without a GPU yet")
def test_hash_bucket_lists(tmpdir):
    df = dispatch._make_df(
        {
            "Authors": [["User_A"], ["User_A", "User_E"], ["User_B", "User_C"], ["User_C"]],
            "Engaging User": ["User_B", "User_B", "User_A", "User_D"],
            "Post": [1, 2, 3, 4],
        }
    )
    cat_names = ["Authors"]  # , "Engaging User"]

    dataset = nvt.Dataset(df)
    hash_features = cat_names >> ops.HashBucket(num_buckets=10)
    processor = nvt.Workflow(hash_features)
    processor.fit(dataset)
    new_gdf = processor.transform(dataset).to_ddf().compute()

    # check to make sure that the same strings are hashed the same
    authors = new_gdf["Authors"].to_arrow().to_pylist()
    assert authors[0][0] == authors[1][0]  # 'User_A'
    assert authors[2][1] == authors[3][0]  # 'User_C'

    assert nvt.ops.get_embedding_sizes(processor)[1]["Authors"][0] == 10
