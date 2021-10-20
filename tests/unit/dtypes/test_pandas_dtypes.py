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

from nvtabular.nvt_dtypes import NVTDtype


@pytest.mark.parametrize(
    "dtype_combo",
    [
        (
            pd.DataFrame(
                {
                    "int": [1, 2, 3, 4],
                    "float": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            ("int", "float"),  # dtype names nvt specific
            True,  # signed
        ),
    ],
)
def test_dtypes(dtype_combo):
    df_pandas, d_names, d_signed = dtype_combo
    pd_dtypes = df_pandas.dtypes
    for dname, dtype in zip(d_names, pd_dtypes):
        nvt_dtype = NVTDtype._from(dtype)
        assert nvt_dtype.name == dname
        assert nvt_dtype.size == dtype.itemsize * 8  # translate to btis
        assert nvt_dtype.signed == d_signed
        assert not nvt_dtype.is_list
        # then convert back for testing
        np_dtype = nvt_dtype._to("pandas")
        assert np_dtype == dtype
