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
def test_drop_low_cardinality(tmpdir, cpu):
    df = pd.DataFrame()
    if not cpu:
        df = cudf.DataFrame(df)

    df["col1"] = ["a", "a", "a", "a", "a"]
    df["col2"] = ["a", "a", "a", "a", "b"]
    df["col3"] = ["a", "a", "b", "b", "c"]

    features = list(df.columns) >> nvt.ops.Categorify() >> nvt.ops.DropLowCardinality()

    workflow = nvt.Workflow(features)
    transformed = workflow.fit_transform(nvt.Dataset(df)).to_ddf().compute()

    assert workflow.output_schema.column_names == ["col2", "col3"]

    expected = df.drop(["col1"], axis=1)
    expected["col2"] = [1, 1, 1, 1, 2]
    expected["col3"] = [1, 1, 2, 2, 3]
    assert_eq(transformed, expected)
