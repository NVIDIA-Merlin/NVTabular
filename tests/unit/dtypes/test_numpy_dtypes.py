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

from nvtabular.nvt_dtypes import NVTDtype


@pytest.mark.parametrize(
    "dtype_combo",
    [
        (np.int, "int", 64, True),
        (np.int8, "int", 8, True),
        (np.int16, "int", 16, True),
        (np.int32, "int", 32, True),
        (np.int64, "int", 64, True),
        (np.uint, "int", 64, False),
        (np.uint8, "int", 8, False),
        (np.uint16, "int", 16, False),
        (np.uint32, "int", 32, False),
        (np.uint64, "int", 64, False),
        (np.float, "float", 64, True),
        (np.float16, "float", 16, True),
        (np.float32, "float", 32, True),
        (np.float64, "float", 64, True),
    ],
)
def test_dtypes(dtype_combo):
    dtype, d_name, d_size, d_signed = dtype_combo
    nvt_dtype = NVTDtype._from(dtype)
    assert nvt_dtype.name == d_name
    assert nvt_dtype.size == d_size
    assert nvt_dtype.signed == d_signed
    assert not nvt_dtype.is_list
    # then convert back for testing
    np_dtype = nvt_dtype._to("numpy")
    assert np_dtype == np.dtype(dtype)
