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
from dataclasses import dataclass
from enum import Enum

# import cudf
import numpy as np
from cudf._typings import Dtype

# import pandas as pd
from cudf.core.dtypes import dtype
from pandas.core.dtypes.dtypes import CategoricalDtype, PandasExtensionDtype
from pandas.core.dtypes.inference import is_nested_list_like


class PandasDtypes:
    @classmethod
    def _to(cls, nvt_dtype):
        # return pandas dtype, input must be nvt.dtype
        if not isinstance(nvt_dtype, NVTDtype):
            raise ValueError(
                f"The supplied dtype({type(nvt_dtype)}) does not extend the NVTDtype class"
            )
        return NumpyDtypes._to(nvt_dtype)

    @classmethod
    def _from(cls, in_dtype):
        # return nvt dtype, input must be pandas dtype or numpy base type
        if isinstance(in_dtype, PandasExtensionDtype):
            # pandas extended routine
            name, signed = cls.dtypes_to_nvt[in_dtype.kind]
            # account for bytes to bits
            size = in_dtype.itemsize * 8
            is_list = bool(is_nested_list_like(in_dtype))
            return NVTDtype(name, size, signed, is_list)
        return NumpyDtypes._from(in_dtype)

    dtypes_to_nvt = {CategoricalDtype: ("object", False)}

    nvt_to_dtypes = {"object": CategoricalDtype}


class CudfDtypes:
    @classmethod
    def _to(cls, nvt_dtype):
        # return cudf dtype, input must be nvt.dtype
        if not isinstance(nvt_dtype, NVTDtype):
            raise ValueError(
                f"The supplied dtype({type(nvt_dtype)}) does not extend the NVTDtype class"
            )
        return dtype(NumpyDtypes._to(nvt_dtype))

    @classmethod
    def _from(cls, in_dtype):
        # return nvt dtype, input must be pandas dtype
        if isinstance(in_dtype, Dtype):
            # pandas extended routine
            name, signed = cls.dtypes_to_nvt[in_dtype.kind]
            # account for bytes to bits
            size = in_dtype.itemsize * 8
            is_list = bool(is_nested_list_like(in_dtype))
            return NVTDtype(name, size, signed, is_list)
        return NumpyDtypes._from(in_dtype)


class NumpyDtypes:
    @classmethod
    def _to(cls, nvt_dtype):
        # return pandas dtype, input must be nvt.dtype
        if not isinstance(nvt_dtype, NVTDtype):
            raise ValueError(
                f"The supplied dtype({type(nvt_dtype)}) does not extend the NVTDtype class"
            )
        signed = "signed" if nvt_dtype.signed else "unsigned"
        return cls.nvt_to_dtypes[nvt_dtype.name][signed][nvt_dtype.size]

    @classmethod
    def _from(cls, in_dtype):
        # return nvt dtype, input must be pandas dtype
        if "numpy" not in str(in_dtype):
            try:
                in_dtype = np.dtype(in_dtype)
            except TypeError as err:
                raise ValueError(f"Cannot convert non supported numpy dtype: {dtype}") from err
        if not isinstance(in_dtype, np.dtype):
            in_dtype = np.dtype(in_dtype)
        # TODO: should be checked against enum strings available
        if in_dtype.kind not in cls.dtypes_to_nvt:
            raise ValueError(
                f"NVTabular does not currently support numpy dtype kind: {in_dtype.kind}"
            )
        name, signed = cls.dtypes_to_nvt[in_dtype.kind]
        # account for bytes to bits
        size = in_dtype.itemsize * 8
        is_list = bool(in_dtype.subdtype)
        return NVTDtype(name, size, signed, is_list)

    # should this use None (aka unsupported) as a third option
    dtypes_to_nvt = {
        "b": ("bool", False),
        "i": ("int", True),
        "u": ("int", False),
        "f": ("float", True),
        "c": ("cfloat", False),
        "m": ("timedelta", False),
        "M": ("datetime", False),
        "O": ("object", False),
        "S": ("byte-string", False),
        "U": ("unicode", False),
        "V": ("void", False),
    }

    nvt_to_dtypes = {
        "int": {
            "signed": {
                8: np.int8,
                16: np.int16,
                32: np.int32,
                64: np.int64,
            },
            "unsigned": {
                8: np.uint8,
                16: np.uint16,
                32: np.uint32,
                64: np.uint64,
            },
        },
        "float": {
            "signed": {
                16: np.float16,
                32: np.float32,
                64: np.float64,
            },
        },
    }


class NVT_Dtypes(Enum):
    INT = "int"
    FLOAT = "float"


@dataclass(frozen=True)
class NVTDtype:
    name: str
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

    @classmethod
    def _from(cls, in_dtype):
        dtype_type = str(in_dtype)
        for dtype_key in list(cls.dtypes_dict.keys()):
            if dtype_key in dtype_type:
                return cls.dtypes_dict[dtype_key]._from(in_dtype)
        try:
            in_dtype = np.dtype(in_dtype)
            return cls.dtypes_dict["numpy"]._from(in_dtype)
        except TypeError as err:
            raise ValueError(
                f"""did not find appropriate dtype base for {in_dtype},
                currently support: {cls.dtypes_dict.keys()}"""
            ) from err
