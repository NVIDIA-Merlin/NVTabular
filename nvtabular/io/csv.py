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
import functools

import dask.dataframe as dd

try:
    import dask_cudf
except ImportError:
    dask_cudf = None
import numpy as np
from dask.bytes import read_bytes
from dask.utils import parse_bytes
from fsspec.core import get_fs_token_paths
from fsspec.utils import infer_compression

from .dataset_engine import DatasetEngine


class CSVDatasetEngine(DatasetEngine):
    """CSVDatasetEngine

    Thin wrapper around dask_cudf.read_csv.
    """

    def __init__(self, paths, part_size, storage_options=None, cpu=False, **kwargs):
        super().__init__(paths, part_size, cpu=cpu, storage_options=storage_options)
        self._meta = {}
        self.csv_kwargs = kwargs
        self.csv_kwargs["storage_options"] = storage_options

        # CSV reader needs a list of files
        # (Assume flat directory structure if this is a dir)
        if len(self.paths) == 1 and self.fs.isdir(self.paths[0]):
            self.paths = self.fs.glob(self.fs.sep.join([self.paths[0], "*"]))

    def to_ddf(self, columns=None, cpu=None):

        # Check if we are using cpu
        cpu = self.cpu if cpu is None else cpu
        if cpu:
            ddf = dd.read_csv(self.paths, blocksize=self.part_size, **self.csv_kwargs)
        else:
            ddf = dask_cudf.read_csv(self.paths, chunksize=self.part_size, **self.csv_kwargs)
        if columns:
            ddf = ddf[columns]
        return ddf

    @property
    @functools.lru_cache(1)
    def _file_partition_map(self):
        ind = 0
        _pp_map = {}
        for path, blocks in zip(
            *_byte_block_counts(
                self.paths,
                self.part_size,
                **self.csv_kwargs,
            )
        ):
            _pp_map[path.split(self.fs.sep)[-1]] = np.arange(ind, ind + blocks)
            ind += blocks
        return _pp_map

    def to_cpu(self):
        self.cpu = True

    def to_gpu(self):
        self.cpu = False


def _byte_block_counts(
    urlpath,
    blocksize,
    lineterminator=None,
    compression="infer",
    storage_options=None,
    **kwargs,
):
    """Return a list of paths and block counts.

    Logic copied from dask.bytes.read_bytes
    """

    if lineterminator is not None and len(lineterminator) == 1:
        kwargs["lineterminator"] = lineterminator
    else:
        lineterminator = "\n"

    if compression == "infer":
        paths = get_fs_token_paths(urlpath, mode="rb", storage_options=storage_options)[2]
        compression = infer_compression(paths[0])

    if isinstance(blocksize, str):
        blocksize = parse_bytes(blocksize)
    if blocksize and compression:
        blocksize = None

    b_out = read_bytes(
        urlpath,
        delimiter=lineterminator.encode(),
        blocksize=blocksize,
        sample=False,
        compression=compression,
        include_path=True,
        **(storage_options or {}),
    )
    _, values, paths = b_out

    if not isinstance(values[0], (tuple, list)):
        values = [values]

    return paths, [len(v) for v in values]
