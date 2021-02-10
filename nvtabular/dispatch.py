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
from typing import Union

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DataFrameType = Union[pd.DataFrame, cudf.DataFrame]


def _arange(size, like_df=None):
    if isinstance(like_df, pd.DataFrame):
        return np.arange(size)
    else:
        return cp.arange(size)


def _series_has_nulls(s):
    if isinstance(s, pd.Series):
        return s.isnull().values.any()
    else:
        return s._column.has_nulls


def _concat_columns(args: list):
    """Dispatch function to concatenate DataFrames with axis=1"""
    if len(args) == 1:
        return args[0]
    else:
        _lib = cudf if isinstance(args[0], cudf.DataFrame) else pd
        return _lib.concat(args, axis=1)
    return None


def _read_parquet_dispatch(df: DataFrameType):
    """Return the necessary read_parquet function to generate
    data of a specified type.
    """
    if isinstance(df, pd.DataFrame):
        return pd.read_parquet
    else:
        return cudf.io.read_parquet


def _parquet_writer_dispatch(df: DataFrameType):
    """Return the necessary ParquetWriter class to write
    data of a specified type.
    """
    if isinstance(df, pd.DataFrame):
        return pq.ParquetWriter
    else:
        return cudf.io.parquet.ParquetWriter
