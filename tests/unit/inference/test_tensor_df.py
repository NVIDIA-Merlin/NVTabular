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
from nvtabular.inference.graph.tesnor_df import TensorDataFrame, TensorSeries


def test_series_create():
    tensor_ser = TensorSeries("test", [1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2])
    assert tensor_ser.name == "test"
    assert tensor_ser.values == [1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2]


def test_subset_series():
    tensor_ser = TensorSeries("test", [1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2])
    new_ser = tensor_ser[1, 2, 3]
    assert new_ser.shape[0] == 3
    assert all(val in new_ser.values for val in [2.0, 4.5, 4.0])


def test_multihot_series():
    pass


def test_multihot_series_subset():
    pass


def test_create_df_multiple_series():
    range_val = 10
    sers = []
    for x in range(range_val):
        sers.append(TensorSeries(f"test_{x}", [1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2]))
    tdf = TensorDataFrame(sers)
    assert len(tdf) == range_val
    for idx, tensor in enumerate(tdf):
        assert str(idx) in tensor.name
        assert tensor.values == [1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2]


def test_subset_selection_tdf():
    range_val = 10
    sers = []
    names = []
    for x in range(range_val):
        name = f"test_{x}"
        sers.append(TensorSeries(name, [1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2]))
        names.append(name)
    tdf = TensorDataFrame(sers)
    new_tdf = tdf[names[:-1]]
    assert len(new_tdf) == len(names[:-1])
    assert all(names[idx] == tensor.name for idx, tensor in enumerate(new_tdf))


def test_create_df_single_hot_offsets():
    range_val = 10
    sers = []
    for x in range(range_val):
        sers.append(TensorSeries(f"test_{x}", [1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2]))
    tdf = TensorDataFrame(sers)
    for tensor in tdf:
        assert tensor.offsets == len([1.0, 2.0, 4.5, 4.0, 6.0, 3.2, 1.2])
