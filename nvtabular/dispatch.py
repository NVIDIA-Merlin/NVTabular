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
import enum
import itertools
from typing import Callable, Union

import cudf
import cupy as cp
import dask.dataframe as dd
import dask_cudf
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from cudf.core.column import as_column, build_column
from cudf.utils.dtypes import is_list_dtype

try:
    # Dask >= 2021.5.1
    from dask.dataframe.core import hash_object_dispatch
except ImportError:
    # Dask < 2021.5.1
    from dask.dataframe.utils import hash_object_dispatch

DataFrameType = Union[pd.DataFrame, cudf.DataFrame]


class ExtData(enum.Enum):
    """Simple Enum to track external-data types"""

    DATASET = 0
    ARROW = 1
    CUDF = 2
    PANDAS = 3
    DASK_CUDF = 4
    DASK_PANDAS = 5
    PARQUET = 6
    CSV = 7


def _is_dataframe_object(x):
    # Simple check if object is a cudf or pandas
    # DataFrame object
    return isinstance(x, (cudf.DataFrame, pd.DataFrame))


def _hex_to_int(s, dtype=None):
    def _pd_convert_hex(x):
        if pd.isnull(x):
            return pd.NA
        return int(x, 16)

    if isinstance(s, cudf.Series):
        # CuDF Version
        if s.dtype == "object":
            s = s.str.htoi()
        return s.astype(dtype or np.int32)
    else:
        # Pandas Version
        if s.dtype == "object":
            s = s.apply(_pd_convert_hex)
        return s.astype("Int64").astype(dtype or "Int32")


def _random_state(seed, like_df=None):
    """Dispatch for numpy.random.RandomState"""
    if isinstance(like_df, (pd.DataFrame, pd.Series)):
        return np.random.RandomState(seed)
    else:
        return cp.random.RandomState(seed)


def _arange(size, like_df=None, dtype=None):
    """Dispatch for numpy.arange"""
    if isinstance(like_df, (pd.DataFrame, pd.Series)):
        return np.arange(size, dtype=dtype)
    else:
        return cp.arange(size, dtype=dtype)


def _array(x, like_df=None, dtype=None):
    """Dispatch for numpy.array"""
    if isinstance(like_df, pd.DataFrame):
        return np.array(x, dtype=dtype)
    else:
        return cp.array(x, dtype=dtype)


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


def _is_list_dtype(ser):
    """Check if Series contains list elements"""
    if isinstance(ser, pd.Series):
        if not len(ser):  # pylint: disable=len-as-condition
            return False
        return pd.api.types.is_list_like(ser.values[0])
    return is_list_dtype(ser)


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


def _read_parquet_dispatch(df: DataFrameType) -> Callable:
    return _read_dispatch(df=df, fmt="parquet")


def _read_dispatch(df: DataFrameType = None, cpu=None, collection=False, fmt="parquet") -> Callable:
    """Return the necessary read_parquet function to generate
    data of a specified type.
    """
    if cpu or isinstance(df, pd.DataFrame):
        _mod = dd if collection else pd
    else:
        _mod = dask_cudf if collection else cudf.io
    _attr = "read_csv" if fmt == "csv" else "read_parquet"
    return getattr(_mod, _attr)


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


def _to_arrow(x):
    """Move data to arrow format"""
    if isinstance(x, cudf.DataFrame):
        return x.to_arrow()
    else:
        return pa.Table.from_pandas(x, preserve_index=False)


def _detect_format(data):
    """Utility to detect the format of `data`"""
    from nvtabular import Dataset

    if isinstance(data, Dataset):
        return ExtData.DATASET
    elif isinstance(data, dd.DataFrame):
        if isinstance(data._meta, cudf.DataFrame):
            return ExtData.DASK_CUDF
        return ExtData.DASK_PANDAS
    elif isinstance(data, cudf.DataFrame):
        return ExtData.CUDF
    elif isinstance(data, pd.DataFrame):
        return ExtData.PANDAS
    elif isinstance(data, pa.Table):
        return ExtData.ARROW
    else:
        mapping = {
            "pq": ExtData.PARQUET,
            "parquet": ExtData.PARQUET,
            "csv": ExtData.CSV,
        }
        if isinstance(data, list) and data:
            file_type = mapping.get(str(data[0]).split(".")[-1], None)
        else:
            file_type = mapping.get(str(data).split(".")[-1], None)
        if file_type is None:
            raise ValueError("Data format not recognized.")
        return file_type


def _convert_data(x, cpu=True, to_collection=None, npartitions=1):
    """Move data between cpu and gpu-backed data.

    Note that the input ``x`` may be an Arrow Table,
    but the output will only be a pandas or cudf DataFrame.
    Use `to_collection=True` to specify that the output should
    always be a Dask collection (otherwise, "serial" DataFrame
    objects will remain "serial").
    """
    if cpu:
        if isinstance(x, dd.DataFrame):
            # If input is a dask_cudf collection, convert
            # to a pandas-backed Dask collection
            if not isinstance(x, dask_cudf.DataFrame):
                # Already a Pandas-backed collection
                return x
            # Convert cudf-backed collection to pandas-backed collection
            return x.to_dask_dataframe()
        else:
            # Make sure _x is a pandas DataFrame
            _x = x if isinstance(x, pd.DataFrame) else x.to_pandas()
            # Output a collection if `to_collection=True`
            return dd.from_pandas(_x, sort=False, npartitions=npartitions) if to_collection else _x
    else:
        if isinstance(x, dd.DataFrame):
            # If input is a Dask collection, covert to dask_cudf
            if isinstance(x, dask_cudf.DataFrame):
                # Already a cudf-backed Dask collection
                return x
            # Convert pandas-backed collection to cudf-backed collection
            return x.map_partitions(cudf.from_pandas)
        elif isinstance(x, pa.Table):
            return cudf.DataFrame.from_arrow(x)
        else:
            # Make sure _x is a cudf DataFrame
            _x = x
            if isinstance(x, pa.Table):
                _x = cudf.DataFrame.from_arrow(x)
            elif isinstance(x, pd.DataFrame):
                _x = cudf.DataFrame.from_pandas(x)
            # Output a collection if `to_collection=True`
            return (
                dask_cudf.from_cudf(_x, sort=False, npartitions=npartitions)
                if to_collection
                else _x
            )


def _to_host(x):
    """Move cudf.DataFrame to host memory for caching.

    All other data will pass through unchanged.
    """
    if isinstance(x, cudf.DataFrame):
        return x.to_arrow()
    else:
        return x
