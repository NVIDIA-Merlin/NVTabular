#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

from merlin.core.compat import cudf
from nvtabular import ColumnSelector, ops
from tests.conftest import assert_eq

if cudf:
    _CPU = [True, False]
else:
    _CPU = [True]


@pytest.mark.parametrize("cpu", _CPU)
def test_list_slice(cpu):
    DataFrame = pd.DataFrame if cpu else cudf.DataFrame

    df = DataFrame({"y": [[0, 1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4]]})

    op = ops.ListSlice(0, 2)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[0, 1], [1, 2], [1, 223]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(3, 5)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[2, 767], [3], []]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(4, 10)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[767], [], []]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(100, 20000)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[], [], []]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-4)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-3, -1)
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[2, 2], [2, 2], [1, 223]]})
    assert_eq(transformed, expected)


@pytest.mark.parametrize("cpu", _CPU)
def test_list_slice_pad(cpu):
    DataFrame = pd.DataFrame if cpu else cudf.DataFrame
    df = DataFrame({"y": [[0, 1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4]]})

    # 0 pad to 5 elements
    op = ops.ListSlice(5, pad=True)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[0, 1, 2, 2, 767], [1, 2, 2, 3, 0], [1, 223, 4, 0, 0]]})
    assert_eq(transformed, expected)

    # make sure we can also pad when start != 0, and when pad_value is set
    op = ops.ListSlice(1, 6, pad=True, pad_value=123)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767, 123], [2, 2, 3, 123, 123], [223, 4, 123, 123, 123]]})
    assert_eq(transformed, expected)

    # we should be able to do pad out negative offsets as well
    op = ops.ListSlice(-4, pad=True, pad_value=-1)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4, -1]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-4, -1, pad=True, pad_value=-1)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2], [1, 2, 2], [1, 223, -1]]})
    assert_eq(transformed, expected)

    op = ops.ListSlice(-4, pad=True, pad_value=-1)
    selector = ColumnSelector(["y"])
    transformed = op.transform(selector, df)
    expected = DataFrame({"y": [[1, 2, 2, 767], [1, 2, 2, 3], [1, 223, 4, -1]]})
    assert_eq(transformed, expected)
