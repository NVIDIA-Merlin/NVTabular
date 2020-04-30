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
import os

import cudf
import pytest

import nvtabular.dl_encoder as encoder
import nvtabular.ds_iterator as ds
from tests.conftest import allcols_csv, mycols_csv


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("use_frequency", [True, False])
def test_dl_encoder_fit_transform_fim(datasets, batch, dskey, use_frequency):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    # create file iterator to go through the
    enc = encoder.DLLabelEncoder("name-string", use_frequency=use_frequency)
    enc.fit(df_expect["name-string"])
    enc.fit_finalize()
    new_ser = enc.transform(df_expect["name-string"])
    unis = set(df_expect["name-string"].tolist())
    assert len(unis) == max(new_ser.tolist())
    for file in enc.file_paths:
        os.remove(file)


@pytest.mark.parametrize("batch", [0, 100, 1000])
@pytest.mark.parametrize("dskey", ["csv", "csv-no-header"])
@pytest.mark.parametrize("use_frequency", [True, False])
def test_dl_encoder_fit_transform_ltm(datasets, batch, dskey, use_frequency):
    paths = glob.glob(str(datasets[dskey]) + "/*.csv")
    names = allcols_csv if dskey == "csv-no-header" else None
    df_expect = cudf.read_csv(paths[0], header=False, names=names)[mycols_csv]
    df_expect["id"] = df_expect["id"].astype("int64")
    data_itr = ds.GPUDatasetIterator(paths[0], batch_size=batch, gpu_memory_frac=2e-8, names=names)
    enc = encoder.DLLabelEncoder(
        "name-string", path=str(datasets["cats"]), limit_frac=1e-10, use_frequency=use_frequency,
    )
    for chunk in data_itr:
        enc.fit(chunk["name-string"])

    enc.fit_finalize()
    new_ser = enc.transform(df_expect["name-string"])
    unis = df_expect["name-string"].unique().values_to_string()
    # set does not pick up None values so must be added if found in
    assert len(unis) == max(new_ser.tolist())
    for file in enc.file_paths:
        os.remove(file)


@pytest.mark.parametrize("values", [[0, 1, 2, 3], ["0", "1", "2", "3"], [0, 1, 2, -1]])
@pytest.mark.parametrize("use_frequency", [True, False])
def test_dl_encoder_off_by_one(values, use_frequency):
    values = cudf.Series(values)
    enc = encoder.DLLabelEncoder("x", use_frequency=use_frequency)
    for _ in range(3):
        enc.fit(values)
    enc.fit_finalize()
    transformed = enc.transform(values)
    assert len(transformed) == len(values)
    assert set(transformed.tolist()) == {1, 2, 3, 4}
