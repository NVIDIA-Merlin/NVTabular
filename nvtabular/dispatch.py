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
import itertools
from typing import Union

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from cudf.core.column import as_column, build_column
from cudf.utils.dtypes import is_list_dtype
from dask.dataframe.utils import hash_object_dispatch

DataFrameType = Union[pd.DataFrame, cudf.DataFrame]


def _arange(size, like_df=None):
    """Dispatch for numpy.arange"""
    if isinstance(like_df, pd.DataFrame):
        return np.arange(size)
    else:
        return cp.arange(size)


def _hash_series(s):
    """Row-wise Series hash"""
    if isinstance(s, pd.Series):
        # Using pandas hashing, which does not produce the
        # same result as cudf.Series.hash_values().  Do not
        # expect hash-based data transformations to be the
        # same on CPU and CPU.  TODO: Fix this (maybe use
        # murmurhash3 manually on CPU).
        return hash_object_dispatch(s).values
    else:
        if _is_list_dtype(s):
            return s.list.leaves.hash_values()
        else:
            return s.hash_values()


def _natural_log(df):
    """Natural logarithm of all columns in a DataFrame"""
    if isinstance(df, pd.DataFrame):
        return pd.DataFrame(np.log(df.values), columns=df.columns, index=df.index)
    else:
        return df.log()


def _series_has_nulls(s):
    """Check if Series contains any null values"""
    if isinstance(s, pd.Series):
        return s.isnull().values.any()
    else:
        return s._column.has_nulls


def _is_list_dtype(s):
    """Check if Series contains list elements"""
    if isinstance(s, pd.Series):
        if not len(s):
            return False
        return pd.api.types.is_list_like(s.values[0])
    else:
        return is_list_dtype(s)


def _flatten_list_column(s):
    """Flatten elements of a list-based column"""
    if isinstance(s, pd.Series):
        return pd.DataFrame({s.name: itertools.chain(*s)})
    else:
        return cudf.DataFrame({s.name: s.list.leaves})


def _concat_columns(args: list):
    """Dispatch function to concatenate DataFrames with axis=1"""
    if len(args) == 1:
        return args[0]
    else:
        _lib = cudf if isinstance(args[0], cudf.DataFrame) else pd
        return _lib.concat(
            [a.reset_index(drop=True) for a in args],
            axis=1,
        )
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


def _encode_list_column(original, encoded):
    """Convert `encoded` to be a list column with the
    same offsets as `original`
    """
    if isinstance(original, pd.Series):
        # Pandas version (not very efficient)
        offset = 0
        new_data = []
        for val in original.values:
            size = len(val)
            new_data.append(list(encoded[offset : offset + size]))
            offset += size
        return pd.Series(new_data)
    else:
        # CuDF version
        encoded = as_column(encoded)
        return build_column(
            None,
            dtype=cudf.core.dtypes.ListDtype(encoded.dtype),
            size=original.size,
            children=(original._column.offsets, encoded),
        )
