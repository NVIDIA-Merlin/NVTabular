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
import gzip
import os
import shutil
import tarfile
import urllib.request
import warnings
import zipfile

import dask
from dask.dataframe.optimize import optimize as dd_optimize
from dask.distributed import get_client
from tqdm import tqdm

try:
    from numba import cuda
except ImportError:
    cuda = None

try:
    import psutil
except ImportError:
    psutil = None


def _pynvml_mem_size(kind="total", index=0):
    import pynvml

    pynvml.nvmlInit()
    size = None
    if kind == "free":
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).free)
    elif kind == "total":
        size = int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).total)
    else:
        raise ValueError("{0} not a supported option for device_mem_size.".format(kind))
    pynvml.nvmlShutdown()
    return size


def device_mem_size(kind="total", cpu=False):

    # Use psutil (if available) for cpu mode
    if cpu and psutil:
        if kind == "total":
            return psutil.virtual_memory().total
        elif kind == "free":
            return psutil.virtual_memory().free
    elif cpu:
        warnings.warn("Please install psutil for full cpu=True support.")
        # Assume 1GB of memory
        return int(1e9)

    if kind not in ["free", "total"]:
        raise ValueError("{0} not a supported option for device_mem_size.".format(kind))
    try:
        if kind == "free":
            return int(cuda.current_context().get_memory_info()[0])
        else:
            return int(cuda.current_context().get_memory_info()[1])
    except NotImplementedError:
        if kind == "free":
            # Not using NVML "free" memory, because it will not include RMM-managed memory
            warnings.warn("get_memory_info is not supported. Using total device memory from NVML.")
        size = _pynvml_mem_size(kind="total", index=0)
    return size


def get_rmm_size(size):
    return (size // 256) * 256


def download_file(url, local_filename, unzip_files=True, redownload=True):
    """utility function to download a dataset file (movielens/criteo/rossmann etc)
    locally, displaying a progress bar during download"""
    local_filename = os.path.abspath(local_filename)
    path = os.path.dirname(local_filename)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if not url.startswith("http"):
        raise ValueError(f"Unhandled url scheme on {url} - this function only is for http")

    if redownload or not os.path.exists(local_filename):
        desc = f"downloading {os.path.basename(local_filename)}"
        with tqdm(unit="B", unit_scale=True, desc=desc) as progress:

            def report(chunk, chunksize, total):
                if not progress.total:
                    progress.reset(total=total)
                progress.update(chunksize)

            opener = urllib.request.build_opener()
            opener.addheaders = [("Accept-Encoding", "gzip, deflate"), ("Accept", "*/*")]

            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, local_filename, reporthook=report)  # nosec

    if unzip_files and local_filename.endswith(".zip"):
        with zipfile.ZipFile(local_filename) as z:
            for filename in tqdm(z.infolist(), desc="unzipping files", unit="files"):
                z.extract(filename, path)

    elif unzip_files and local_filename.endswith(".tgz"):
        with tarfile.open(local_filename, "r") as tar:
            for filename in tqdm(tar.getnames(), desc="untarring files", unit="files"):
                tar.extract(filename, path)

    elif unzip_files and local_filename.endswith(".gz"):
        with gzip.open(local_filename, "rb") as input_file:
            with open(local_filename[:-3], "wb") as output_file:
                shutil.copyfileobj(input_file, output_file)


def _ensure_optimize_dataframe_graph(ddf=None, dsk=None, keys=None):
    # Perform HLG DataFrame optimizations
    #
    # If `ddf` is specified, an optimized Dataframe
    # collection will be returned. If `dsk` and `keys`
    # are specified, an optimized graph will be returned.
    #
    # These optimizations are performed automatically
    # when a DataFrame collection is computed/persisted,
    # but they are NOT always performed when statistics
    # are computed. The purpose of this utility is to
    # ensure that the Dataframe-based optimizations are
    # always applied.

    if ddf is None:
        if dsk is None or keys is None:
            raise ValueError("Must specify both `dsk` and `keys` if `ddf` is not supplied.")
    dsk = ddf.dask if dsk is None else dsk
    keys = ddf.__dask_keys__() if keys is None else keys

    if isinstance(dsk, dask.highlevelgraph.HighLevelGraph):
        with dask.config.set({"optimization.fuse.active": False}):
            dsk = dd_optimize(dsk, keys=keys)

    if ddf is None:
        # Return optimized graph
        return dsk

    # Return optimized ddf
    ddf.dask = dsk
    return ddf


def global_dask_client(client):
    # Check for a global Dask client
    # (if `client` is not already set)
    if not client:
        try:
            return get_client()
        except ValueError:
            return None
    return None


def run_on_worker(func, *args, **kwargs):
    # Run a function on a Dask worker using `delayed`
    # execution (if a Dask client is detected)
    if global_dask_client(kwargs.get("client", None)):
        # There is a specified or global Dask client. Use it
        return dask.delayed(func)(*args, **kwargs).compute()
    # No Dask client - Use simple function call
    return func(*args, **kwargs)
