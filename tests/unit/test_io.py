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

import glob

import cudf
import pytest

import nvtabular.io
from nvtabular.dask.io import DaskDataset
from tests.conftest import allcols_csv, mycols_csv, mycols_pq


@pytest.mark.parametrize("engine", ["csv", "parquet", "csv-no-header"])
def test_shuffle_gpu(tmpdir, datasets, engine):
    num_files = 2
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    shuf = nvtabular.io.Shuffler(tmpdir, num_files)
    shuf.add_data(df1)
    writer_files = shuf.writer_files
    shuf.close()
    if engine == "parquet":
        df3 = cudf.read_parquet(writer_files[0])[mycols_pq]
        df4 = cudf.read_parquet(writer_files[1])[mycols_pq]
    else:
        df3 = cudf.read_parquet(writer_files[0])[mycols_csv]
        df4 = cudf.read_parquet(writer_files[1])[mycols_csv]
    assert df1.shape[0] == df3.shape[0] + df4.shape[0]


@pytest.mark.parametrize("gpu_memory_frac", [0.01, 0.1])
@pytest.mark.parametrize("engine", ["csv", "parquet"])
def test_dask_dataset(tmpdir, datasets, engine, gpu_memory_frac):
    paths = glob.glob(str(datasets[engine]) + "/*." + engine.split("-")[0])
    if engine == "parquet":
        df1 = cudf.read_parquet(paths[0])[mycols_pq]
    else:
        df1 = cudf.read_csv(paths[0], header=False, names=allcols_csv)[mycols_csv]
    if engine == "parquet":
        columns = mycols_pq
    else:
        columns = mycols_csv

    dd = DaskDataset(paths[0], engine=engine, part_mem_fraction=gpu_memory_frac, names=columns)
    size = 0
    for chunk in dd:
        size += chunk.shape[0]

    assert size == df1.shape[0]
