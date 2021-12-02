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
import functools
import itertools
from typing import Callable, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import cudf
    import cupy as cp
    import dask_cudf
    import rmm
    from cudf.core.column import as_column, build_column

    try:
        # cudf >= 21.08
        from cudf.api.types import is_list_dtype, is_string_dtype
    except ImportError:
        # cudf < 21.08
        from cudf.utils.dtypes import is_list_dtype, is_string_dtype

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cudf = None
    rmm = None

try:
    # Dask >= 2021.5.1
    from dask.dataframe.core import hash_object_dispatch
except ImportError:
    # Dask < 2021.5.1
    from dask.dataframe.utils import hash_object_dispatch

try:
    import nvtx

    annotate = nvtx.annotate
except ImportError:
    # don't have nvtx installed - don't annotate our functions
    def annotate(*args, **kwargs):
        def inner1(func):
            @functools.wraps(func)
            def inner2(*args, **kwargs):
                return func(*args, **kwargs)

            return inner2

        return inner1


if HAS_GPU:
    DataFrameType = Union[pd.DataFrame, cudf.DataFrame]
    SeriesType = Union[pd.Series, cudf.Series]
else:
    DataFrameType = Union[pd.DataFrame]
    SeriesType = Union[pd.Series]

# Define mapping between non-nullable,
# and nullable types in Pandas
_PD_NULLABLE_MAP = {
    "int32": "Int32",
    "int64": "Int64",
}


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


def _create_nvt_dataset(df):
    from nvtabular import Dataset

    if not isinstance(df, Dataset):
        # turn arrow format into readable for dispatch
        df_ext_format = _detect_format(df)
        if df_ext_format == ExtData.ARROW:
            df = df.to_pandas() if not cudf else cudf.DataFrame.from_arrow(df)
            # run through make df to safely cast to df
        elif df_ext_format in [ExtData.DASK_CUDF, ExtData.DASK_PANDAS]:
            df = df.compute()
        return Dataset(df)
    return df


def _read_parquet_metadata(path):
    if HAS_GPU:
        return cudf.io.read_parquet_metadata(path)
    full_meta = pq.read_metadata(path)
    pf = pq.ParquetFile(path)
    return full_meta.num_rows, full_meta.num_row_groups, pf.schema.names


def get_lib():
    return cudf if HAS_GPU else pd


def reinitialize(managed_memory=False):
    if rmm:
        rmm.reinitialize(managed_memory=managed_memory)
    return


def random_uniform(size):
    """Dispatch for numpy.random.RandomState"""
    if HAS_GPU:
        return cp.random.uniform(size=size)
    else:
        return np.random.uniform(size=size)


def coo_matrix(data, row, col):
    if HAS_GPU:
        return cp.sparse.coo_matrix((data, row, col))
    else:
        import scipy

        return scipy.sparse.coo_matrix((data, row, col))


def _is_dataframe_object(x):
    # Simple check if object is a cudf or pandas
    # DataFrame object
    if not HAS_GPU:
        return isinstance(x, pd.DataFrame)
    return isinstance(x, (cudf.DataFrame, pd.DataFrame))


def _nullable_series(data, like_df, dtype):
    # Return a Series containing the elements in `data`,
    # with a nullable version of `dtype`, using `like_df`
    # to infer the Series constructor
    if isinstance(like_df, pd.DataFrame):
        # Note that we cannot use "int32"/"int64" to
        # represent nullable data in pandas
        return like_df._constructor_sliced(data, dtype=_PD_NULLABLE_MAP.get(str(dtype), dtype))
    return like_df._constructor_sliced(data, dtype=dtype)


def _is_series_object(x):
    # Simple check if object is a cudf or pandas
    # Series object
    if not HAS_GPU:
        return isinstance(x, pd.Series)
    return isinstance(x, (cudf.Series, pd.Series))


def _is_cpu_object(x):
    # Simple check if object is a cudf or pandas
    # DataFrame object
    return isinstance(x, (pd.DataFrame, pd.Series))


def is_series_or_dataframe_object(maybe_series_or_df):
    return _is_series_object(maybe_series_or_df) or _is_dataframe_object(maybe_series_or_df)


def _hex_to_int(s, dtype=None):
    def _pd_convert_hex(x):
        if pd.isnull(x):
            return pd.NA
        return int(x, 16)

    if isinstance(s, pd.Series):
        # Pandas Version
        if s.dtype == "object":
            s = s.apply(_pd_convert_hex)
        return s.astype("Int64").astype(dtype or "Int32")
    else:
        # CuDF Version
        if s.dtype == "object":
            s = s.str.htoi()
        return s.astype(dtype or np.int32)


def _random_state(seed, like_df=None):
    """Dispatch for numpy.random.RandomState"""
    if not HAS_GPU or isinstance(like_df, (pd.DataFrame, pd.Series)):
        return np.random.RandomState(seed)
    else:
        return cp.random.RandomState(seed)


def _arange(size, like_df=None, dtype=None):
    """Dispatch for numpy.arange"""
    if not HAS_GPU or isinstance(like_df, (np.ndarray, pd.DataFrame, pd.Series)):
        return np.arange(size, dtype=dtype)
    else:
        return cp.arange(size, dtype=dtype)


def _array(x, like_df=None, dtype=None):
    """Dispatch for numpy.array"""
    if not HAS_GPU or isinstance(like_df, (np.ndarray, pd.DataFrame, pd.Series)):
        return np.array(x, dtype=dtype)
    else:
        return cp.array(x, dtype=dtype)


def _zeros(size, like_df=None, dtype=None):
    """Dispatch for numpy.array"""
    if not HAS_GPU or isinstance(like_df, (np.ndarray, pd.DataFrame, pd.Series)):
        return np.zeros(size, dtype=dtype)
    else:
        return cp.zeros(size, dtype=dtype)


def _hash_series(ser):
    """Row-wise Series hash"""
    if not HAS_GPU or isinstance(ser, pd.Series):
        # Using pandas hashing, which does not produce the
        # same result as cudf.Series.hash_values().  Do not
        # expect hash-based data transformations to be the
        # same on CPU and CPU.  TODO: Fix this (maybe use
        # murmurhash3 manually on CPU).
        return hash_object_dispatch(ser).values
    else:
        if _is_list_dtype(ser):
            return ser.list.leaves.hash_values()
        else:
            return ser.hash_values()


def _series_has_nulls(s):
    """Check if Series contains any null values"""
    if isinstance(s, pd.Series):
        return s.isnull().values.any()
    else:
        return s._column.has_nulls


def _list_val_dtype(ser):
    if _is_list_dtype(ser):
        if HAS_GPU and isinstance(ser, cudf.Series):
            return ser.dtype._typ.value_type.to_pandas_dtype()
        elif isinstance(ser, pd.Series):
            return type(ser[0][0])
    return None


def _is_list_dtype(ser):
    """Check if Series contains list elements"""
    if isinstance(ser, pd.Series):
        if not len(ser):  # pylint: disable=len-as-condition
            return False
        return pd.api.types.is_list_like(ser.values[0])
    elif not HAS_GPU:
        # either np.ndarray or a dtype
        if isinstance(ser, np.ndarray):
            ser = ser[0]
        return pd.api.types.is_list_like(np.dtype(ser))
    return is_list_dtype(ser)


def _is_string_dtype(obj):
    if not HAS_GPU:
        return pd.api.types.is_string_dtype(obj)
    else:
        return is_string_dtype(obj)


def _flatten_list_column_values(s):
    """returns a flattened list from a list column"""
    if isinstance(s, pd.Series) or not cudf:
        return pd.Series(itertools.chain(*s))
    else:
        return s.list.leaves


def _flatten_list_column(s):
    """Flatten elements of a list-based column, and return as a DataFrame"""
    values = _flatten_list_column_values(s)
    if isinstance(s, pd.Series) or not cudf:
        return pd.DataFrame({s.name: values})
    else:
        return cudf.DataFrame({s.name: values})


def _concat_columns(args: list):
    """Dispatch function to concatenate DataFrames with axis=1"""
    if len(args) == 1:
        return args[0]
    else:
        _lib = cudf if HAS_GPU and isinstance(args[0], cudf.DataFrame) else pd
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
    if cpu or isinstance(df, pd.DataFrame) or not HAS_GPU:
        _mod = dd if collection else pd
    else:
        _mod = dask_cudf if collection else cudf.io
    _attr = "read_csv" if fmt == "csv" else "read_parquet"
    return getattr(_mod, _attr)


def _parquet_writer_dispatch(df: DataFrameType, path=None, **kwargs):
    """Return the necessary ParquetWriter class to write
    data of a specified type.

    If `path` is specified, an initialized `ParquetWriter`
    object will be returned.  To do this, the pyarrow schema
    will be inferred from df, and kwargs will be used for the
    ParquetWriter-initialization call.
    """
    _args = []
    if isinstance(df, pd.DataFrame):
        _cls = pq.ParquetWriter
        if path:
            _args.append(pa.Table.from_pandas(df, preserve_index=False).schema)
    else:
        _cls = cudf.io.parquet.ParquetWriter

    if not path:
        return _cls

    ret = _cls(path, *_args, **kwargs)
    if isinstance(df, pd.DataFrame):
        ret.write_table = lambda df: _cls.write_table(
            ret, pa.Table.from_pandas(df, preserve_index=False)
        )
    return ret


def _encode_list_column(original, encoded, dtype=None):
    """Convert `encoded` to be a list column with the
    same offsets as `original`
    """
    if isinstance(original, pd.Series):
        # Pandas version (not very efficient)
        offset = 0
        new_data = []
        for val in original.values:
            size = len(val)
            new_data.append(np.array(encoded[offset : offset + size], dtype=dtype))
            offset += size
        return pd.Series(new_data)
    else:
        # CuDF version
        encoded = as_column(encoded)
        if dtype:
            encoded = encoded.astype(dtype, copy=False)
        list_dtype = cudf.core.dtypes.ListDtype(encoded.dtype if dtype is None else dtype)
        return build_column(
            None,
            dtype=list_dtype,
            size=original.size,
            children=(original._column.offsets, encoded),
        )


def _pull_apart_list(original):
    values = _flatten_list_column_values(original)
    if isinstance(original, pd.Series):
        offsets = pd.Series([0]).append(original.map(len).cumsum()).reset_index(drop=True)
        if isinstance(offsets[0], list):
            offsets = pd.Series(offsets.reshape().flatten()).reset_index(drop=True)
    else:
        offsets = original._column.offsets
        elements = original._column.elements
        if isinstance(elements, cudf.core.column.lists.ListColumn):
            offsets = elements.offsets[offsets]
    return _make_series(values), _make_series(offsets)


def _to_arrow(x):
    """Move data to arrow format"""
    if isinstance(x, pd.DataFrame):
        return pa.Table.from_pandas(x, preserve_index=False)
    else:
        return x.to_arrow()


def _concat(objs, **kwargs):
    if isinstance(objs[0], dd.DataFrame):
        return dd.multi.concat(objs)
    elif isinstance(objs[0], (pd.DataFrame, pd.Series)) or not HAS_GPU:
        return pd.concat(objs, **kwargs)
    else:
        return cudf.core.reshape.concat(objs, **kwargs)


def _make_df(_like_df=None, device=None):
    if not cudf or isinstance(_like_df, (pd.DataFrame, pd.Series)):
        return pd.DataFrame(_like_df)
    elif isinstance(_like_df, (cudf.DataFrame, cudf.Series)):
        return cudf.DataFrame(_like_df)
    elif device is None and isinstance(_like_df, dict) and len(_like_df) > 0:
        is_pandas = all(isinstance(v, pd.Series) for v in _like_df.values())
        return pd.DataFrame(_like_df) if is_pandas else cudf.DataFrame(_like_df)
    if device == "cpu":
        return pd.DataFrame(_like_df)
    return cudf.DataFrame(_like_df)


def _make_series(_like_ser=None, device=None):
    if not cudf or device == "cpu":
        return pd.Series(_like_ser)
    return cudf.Series(_like_ser)


def _add_to_series(series, to_add, prepend=True):
    if isinstance(series, pd.Series):
        series_to_add = pd.Series(to_add)
    elif isinstance(series, cudf.Series):
        series_to_add = cudf.Series(to_add)
    else:
        raise ValueError("Unrecognized series, please provide either a pandas a cudf series")

    series_to_concat = [series_to_add, series] if prepend else [series, series_to_add]

    return _concat(series_to_concat)


def _detect_format(data):
    """Utility to detect the format of `data`"""
    from nvtabular import Dataset

    if isinstance(data, Dataset):
        return ExtData.DATASET
    elif isinstance(data, dd.DataFrame):
        if isinstance(data._meta, pd.DataFrame):
            return ExtData.DASK_PANDAS
        return ExtData.DASK_CUDF
    elif isinstance(data, pd.DataFrame):
        return ExtData.PANDAS
    elif isinstance(data, pa.Table):
        return ExtData.ARROW
    elif isinstance(data, cudf.DataFrame):
        return ExtData.CUDF
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
            if cudf is None or not isinstance(x, dask_cudf.DataFrame):
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
            # If input is a Dask collection, convert to dask_cudf
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
    if not HAS_GPU or isinstance(x, (pd.DataFrame, dd.DataFrame)):
        return x
    else:
        return x.to_arrow()


def _from_host(x):
    if not HAS_GPU:
        return x
    elif isinstance(x, cudf.DataFrame):
        return x
    else:
        return cudf.DataFrame.from_arrow(x)


def _build_cudf_list_column(new_elements, new_offsets):
    if not HAS_GPU:
        return []
    return build_column(
        None,
        dtype=cudf.core.dtypes.ListDtype(new_elements.dtype),
        size=new_offsets.size - 1,
        children=(as_column(new_offsets), as_column(new_elements)),
    )


def create_multihot_col(offsets, elements):
    """
    offsets = cudf series with offset values for list data
    data = cudf series with the list data flattened to 1-d
    """
    if isinstance(elements, pd.Series):
        col = pd.Series()
        lh, rh = pd.Series(offsets[1:]).reset_index(drop=True), pd.Series(offsets[:-1]).reset_index(
            drop=True
        )
        vals_per_entry = lh - rh
        vals_used = 0
        entries = []
        for vals_count in vals_per_entry:
            vals_count = int(vals_count)
            entry = elements[vals_used : vals_used + vals_count]
            if len(entry) == 1:
                entry = entry[0]
            vals_used += vals_count
            entries.append(entry.values)
        col = col.append(pd.Series(entries))
    else:
        offsets = as_column(offsets, dtype="int32")
        elements = as_column(elements)
        col = _build_cudf_list_column(elements, offsets)
        col = cudf.Series(col)
    return col


def _generate_local_seed(global_rank, global_size):
    random_state = _get_random_state()
    if cp:
        seeds = random_state.tomaxint(size=global_size)
        cp.random.seed(seeds[global_rank].get())
    else:
        seeds = random_state.randint(0, 2 ** 32, size=global_size)
    return seeds[global_rank]


def _get_random_state():
    if cp:
        return cp.random.get_random_state()
    return np.random.mtrand.RandomState()
