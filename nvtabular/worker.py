#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import contextlib
import threading
from io import BytesIO

import cudf
from dask.distributed import get_worker

# Use global variable as the default
# cache when there are no distributed workers.
# Use a thread lock to "handle" multiple Dask
# worker threads.
_WORKER_CACHE = {}
_WORKER_CACHE_LOCK = threading.RLock()


@contextlib.contextmanager
def get_worker_cache(name):
    with _WORKER_CACHE_LOCK:
        yield _get_worker_cache(name)


def _get_worker_cache(name):
    """Utility to get the `name` element of the cache
    dictionary for the current worker.  If executed
    by anything other than a distributed Dask worker,
    we will use the global `_WORKER_CACHE` variable.
    """
    try:
        worker = get_worker()
    except ValueError:
        # There is no dask.distributed worker.
        # Assume client/worker are same process
        global _WORKER_CACHE
        if name not in _WORKER_CACHE:
            _WORKER_CACHE[name] = {}
        return _WORKER_CACHE[name]
    if not hasattr(worker, "worker_cache"):
        worker.worker_cache = {}
    if name not in worker.worker_cache:
        worker.worker_cache[name] = {}
    return worker.worker_cache[name]


def fetch_table_data(
    table_cache, path, cache="disk", cats_only=False, reader=None, columns=None, **kwargs
):
    """Utility to retrieve a cudf DataFrame from a cache (and add the
    DataFrame to a cache if the element is missing).  Note that `cats_only=True`
    results in optimized logic for the `Categorify` transformation.
    """
    table = table_cache.get(path, None)
    if table and not isinstance(table, cudf.DataFrame):
        if not cats_only:
            return cudf.io.read_parquet(table, index=False)
        df = cudf.io.read_parquet(table, index=False, columns=columns)
        df.index.name = "labels"
        df.reset_index(drop=False, inplace=True)
        return df

    reader = reader or cudf.io.read_parquet
    if table is None:
        if cache in ("device", "disk"):
            table = reader(path, index=False, columns=columns, **kwargs)
        elif cache == "host":
            if reader == cudf.io.read_parquet:
                # If the file is already in parquet format,
                # we can just move the same bytes to host memory
                with open(path, "rb") as f:
                    table_cache[path] = BytesIO(f.read())
                table = reader(table_cache[path], index=False, columns=columns, **kwargs)
            else:
                # Otherwise, we should convert the format to parquet
                table = reader(path, index=False, columns=columns, **kwargs)
                table_cache[path] = BytesIO()
                table.to_parquet(table_cache[path])
        if cats_only:
            table.index.name = "labels"
            table.reset_index(drop=False, inplace=True)
        if cache == "device":
            table_cache[path] = table.copy(deep=False)
    return table


def clean_worker_cache(name=None):
    """Utility to clean the cache dictionary for the
    current worker.  If a `name` argument is passed,
    only that element of the dictionary will be removed.
    """
    with _WORKER_CACHE_LOCK:
        try:
            worker = get_worker()
        except ValueError:
            global _WORKER_CACHE
            if _WORKER_CACHE != {}:
                if name:
                    del _WORKER_CACHE[name]
                else:
                    del _WORKER_CACHE
                    _WORKER_CACHE = {}
            return
        if hasattr(worker, "worker_cache"):
            if name:
                del worker.worker_cache[name]
            else:
                del worker.worker_cache
        return
