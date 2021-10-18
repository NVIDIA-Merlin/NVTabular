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
import numpy as np

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
        if not isinstance(dtype, NVTDtype):
            raise ValueError(
                f"The supplied dtype({type(dtype)}) does not extend the NVTDtype class"
            )
        signed = "signed" if dtype.signed else "unsigned"
        return cls.nvt_to_dtypes[dtype.name][signed][dtype.size]

    @classmethod
    def _from(cls, dtype):
        # return nvt dtype, input must be pandas dtype
        if not isinstance(dtype, np.dtype):
            raise ValueError(f"Cannot convert non support numpy dtype: {type(dtype)}")
        if dtype.kind not in cls.dtypes_to_nvt:
            raise ValueError(f"NVTabular does not currently support numpy dtype kind: {dtype.kind}")
        name = cls.dtypes_to_nvt[dtype.kind]
        size = dtype.itemsize
        signed = bool(name == "signed")
        is_list = bool(dtype.subdtype)
        return NVTDtype(name, size, signed, is_list)

    dtypes_to_nvt = {
        "b": "bool",
        "i": "signed",
        "u": "unsigned",
        "f": "float",
        "c": "cfloat",
        "m": "timedelta",
        "M": "datetime",
        "O": "object",
        "S": "byte-string",
        "U": "unicode",
        "V": "void",
    }

    nvt_to_dtypes = {
        "int": {
            "signed": {
                0: np.int,
                8: np.int8,
                16: np.int16,
                32: np.int32,
                64: np.int64,
            },
            "unsigned": {
                0: np.uint,
                8: np.uint8,
                16: np.uint16,
                32: np.uint32,
                64: np.uint64,
            },
        },
        "float": {
            0: np.float,
            16: np.float16,
            32: np.float32,
            64: np.float64,
        },
    }


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
