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

import cudf

# import cudf
import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype

from nvtabular.dispatch import _is_list_dtype


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
        if isinstance(in_dtype, pd.core.dtypes.base.ExtensionDtype):
            # pandas extended routine
            name, signed = cls.dtypes_to_nvt[in_dtype.kind]
            # account for bytes to bits
            size = in_dtype.itemsize * 8
            is_list = bool(_is_list_dtype(in_dtype))
            return NVTDtype(name, size, signed, is_list)
        return NumpyDtypes._from(in_dtype)

    dtypes_to_nvt = {CategoricalDtype: ("object", False)}

    nvt_to_dtypes = {"object": CategoricalDtype}

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


class CudfDtypes:
    @classmethod
    def _to(cls, nvt_dtype):
        # return cudf dtype, input must be nvt.dtype
        if not isinstance(nvt_dtype, NVTDtype):
            raise ValueError(
                f"The supplied dtype({type(nvt_dtype)}) does not extend the NVTDtype class"
            )
        return NumpyDtypes._to(nvt_dtype)

    @classmethod
    def _from(cls, in_dtype):
        # return nvt dtype, input must be pandas dtype
        if isinstance(in_dtype, cudf.core.dtypes._BaseDtype):
            # pandas extended routine
            name, signed = cls.dtypes_to_nvt[in_dtype.kind]
            # account for bytes to bits
            size = None
            if hasattr(in_dtype, "itemsize"):
                size = in_dtype.itemsize * 8
            is_list = isinstance(in_dtype, cudf.core.dtypes.ListDtype)
            return NVTDtype(name, size, signed, is_list)
        return NumpyDtypes._from(in_dtype)

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

    nvt_to_dtypes = {}


class NumpyDtypes:
    @classmethod
    def _to(cls, nvt_dtype):
        # return numpy dtype, input must be nvt.dtype
        if not isinstance(nvt_dtype, NVTDtype):
            raise ValueError(
                f"The supplied dtype({type(nvt_dtype)}) does not extend the NVTDtype class"
            )
        signed = "signed" if nvt_dtype.signed else "unsigned"
        return cls.nvt_to_dtypes[nvt_dtype.name][signed][nvt_dtype.size]

    @classmethod
    def _from(cls, in_dtype):
        # return nvt dtype, input must be pandas dtype
        if "numpy" not in str(type(in_dtype)):
            try:
                in_dtype = np.dtype(in_dtype)
            except TypeError as err:
                raise ValueError(f"Cannot convert non supported numpy dtype: {in_dtype}") from err
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
                # 64: np.int,
                8: np.int8,
                16: np.int16,
                32: np.int32,
                64: np.int64,
            },
            "unsigned": {
                # 64: np.uint,
                8: np.uint8,
                16: np.uint16,
                32: np.uint32,
                64: np.uint64,
            },
        },
        "float": {
            "signed": {
                # 64: np.float,
                16: np.float16,
                32: np.float32,
                64: np.float64,
            },
        },
        "datetime": {
            "unsigned": {64: np.datetime64},
        },
        "object": {"unsigned": {64: str}},
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
    dtypes_dict = {"numpy": NumpyDtypes, "cudf": CudfDtypes, "pandas": PandasDtypes}

    def _to(self, target_framework):
        if target_framework in self.dtypes_dict:
            return self.dtypes_dict[target_framework]._to(self)
        raise ValueError(
            f"did not find appropriate framework, currently support: {self.dtypes_dict.keys()}"
        )

    @classmethod
    def _from(cls, in_dtype):
        dtype_type = str(type(in_dtype))
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


# def cudf_dtype(arbitrary):
#     """
#     Return the cuDF-supported dtype corresponding to `arbitrary`.
#     Parameters
#     ----------
#     arbitrary: dtype or scalar-like
#     Returns
#     -------
#     dtype: the cuDF-supported dtype that best matches `arbitrary`
#     """
#     # first, try interpreting arbitrary as a NumPy dtype that we support:
#     try:
#         np_dtype = np.dtype(arbitrary)
#         if np_dtype.kind in ("OU"):
#             return np.dtype("object")
#     except TypeError:
#         pass
#     else:
#         if np_dtype not in cudf._lib.types.SUPPORTED_NUMPY_TO_LIBCUDF_TYPES:
#             raise TypeError(f"Unsupported type {np_dtype}")
#         return np_dtype

#     #  next, check if `arbitrary` is one of our extension types:
#     if isinstance(arbitrary, cudf.core.dtypes._BaseDtype):
#         return arbitrary

#     # use `pandas_dtype` to try and interpret
#     # `arbitrary` as a Pandas extension type.
#     #  Return the corresponding NumPy/cuDF type.
#     pd_dtype = pd.api.types.pandas_dtype(arbitrary)
#     try:
#         return cudf_dtype(pd_dtype.numpy_dtype)
#     except AttributeError:
#         if isinstance(pd_dtype, pd.CategoricalDtype):
#             return cudf.CategoricalDtype.from_pandas(pd_dtype)
#         elif isinstance(pd_dtype, pd.StringDtype):
#             return np.dtype("object")
#         elif isinstance(pd_dtype, pd.IntervalDtype):
#             return cudf.IntervalDtype.from_pandas(pd_dtype)
#         else:
#             raise TypeError(
#                 f"Cannot interpret {arbitrary} as a valid cuDF dtype"
#             )
