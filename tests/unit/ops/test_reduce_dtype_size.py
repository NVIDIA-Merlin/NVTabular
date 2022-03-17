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
import pandas as pd
import pytest

import nvtabular as nvt
from tests.conftest import assert_eq

try:
    import cudf

    _CPU = [True, False]
except ImportError:
    _CPU = [True]


@pytest.mark.parametrize("cpu", _CPU)
def test_reduce_size(tmpdir, cpu):
    df = pd.DataFrame()
    if not cpu:
        df = cudf.DataFrame(df)

    df["int16"] = np.array([2 ** 15 - 1, 0], dtype="int64")
    df["int32"] = np.array([2 ** 30, -(2 ** 30)], dtype="int64")
    df["int64"] = np.array([2 ** 60, -(2 ** 60)], dtype="int64")
    df["float32"] = np.array([1.0, 2.0], dtype="float64")

    workflow = nvt.Workflow(list(df.columns) >> nvt.ops.ReduceDtypeSize())
    transformed = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute()

    expected = df
    for column in df:
        expected[column] = expected[column].astype(column)

    assert_eq(expected, transformed)
