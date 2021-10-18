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
import inspect
from dataclasses import dataclass
from enum import Enum

# import cudf
# import numpy as np
# import pandas as pd


class PandasDtypes:
    @classmethod
    def _to(cls, dtype):
        # return pandas dtype, input must be nvt.dtype
        pass

    @classmethod
    def _from(cls, dtype):
        # return nvt dtype, input must be pandas dtype
        pass

    dtypes_dict = {}


class CudfDtypes:
    @classmethod
    def _to(cls, dtype):
        # return pandas dtype, input must be nvt.dtype
        pass

    @classmethod
    def _from(cls, dtype):
        # return nvt dtype, input must be pandas dtype
        pass

    dtypes_dict = {}


class NumpyDtypes:
    @classmethod
    def _to(cls, dtype):
        # return pandas dtype, input must be nvt.dtype
        pass

    @classmethod
    def _from(cls, dtype):
        # return nvt dtype, input must be pandas dtype
        pass

    dtypes_dict = {}


class NVT_Dtypes(Enum):
    INT = ("int",)
    FLOAT = ("float",)


@dataclass(frozen=True)
class NVTDtype:
    dtype: str
    size: int = 0
    signed: bool = False
    is_list: bool = False
    dtypes_dict = {
        "pandas": PandasDtypes,
        "numpy": NumpyDtypes,
        "cudf": CudfDtypes,
    }

    def _to(self, target_framework):
        if target_framework in self.dtypes_dict:
            return self.dtypes_dict[target_framework]._to(self)
        raise ValueError(
            f"did not find appropriate framework, currently support: {self.dtypes_dict.keys()}"
        )

    def _from(self, dtype):
        dtype_type = inspect.getmembers(dtype, inspect.isclass)[0][1]
        dtype_type = str(dtype_type)
        for dtype_key in list(self.dtypes_dict.keys()):
            if dtype_key in dtype_type:
                return self.dtypes_dict[dtype_type]._from(dtype)
        raise ValueError(
            f"did not find appropriate dtype base, currently support: {self.dtypes_dict.keys()}"
        )
