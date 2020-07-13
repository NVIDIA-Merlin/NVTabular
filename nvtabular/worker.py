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

from dask.distributed import get_worker

# Use global variable as the default
# cache when there are no distributed workers
WORKER_CACHE = {}


def get_worker_cache(name):
    """ Utility to get the `name` element of the cache
        dictionary for the current worker.  If executed
        by anything other than a distributed Dask worker,
        we will use the global `WORKER_CACHE` variable.
    """
    try:
        worker = get_worker()
    except ValueError:
        # There is no dask.distributed worker.
        # Assume client/worker are same process
        global WORKER_CACHE
        if name not in WORKER_CACHE:
            WORKER_CACHE[name] = {}
        return WORKER_CACHE[name]
    if not hasattr(worker, "worker_cache"):
        worker.worker_cache = {}
    if name not in worker.worker_cache:
        worker.worker_cache[name] = {}
    return worker.worker_cache[name]


def clean_worker_cache(name=None):
    """ Utility to clean the cache dictionary for the
        current worker.  If a `name` argument is passed,
        only that element of the dictionary will be removed.
    """
    try:
        worker = get_worker()
    except ValueError:
        global WORKER_CACHE
        if WORKER_CACHE != {}:
            if name:
                del WORKER_CACHE[name]
            else:
                del WORKER_CACHE
                WORKER_CACHE = {}
        return
    if hasattr(worker, "worker_cache"):
        if name:
            del worker.worker_cache[name]
        else:
            del worker.worker_cache
    return
