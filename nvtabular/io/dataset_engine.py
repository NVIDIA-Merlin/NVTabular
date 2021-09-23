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

from dask.utils import natural_sort_key
from fsspec.core import get_fs_token_paths


class DatasetEngine:
    """Base class for Dask-powered IO engines. Engines must provide a ``to_ddf`` method."""

    def __init__(self, paths, part_size, cpu=False, storage_options=None):
        paths = sorted(paths, key=natural_sort_key)
        self.paths = paths
        self.part_size = part_size
        self.storage_options = storage_options
        fs, fs_token, paths2 = get_fs_token_paths(
            paths, mode="rb", storage_options=self.storage_options
        )
        self.stripped_paths = paths2
        self.fs = fs
        self.fs_token = fs_token
        self.cpu = cpu

    def to_ddf(self, columns=None, cpu=None):
        raise NotImplementedError(""" Return a dask.dataframe.DataFrame or dask_cudf.DataFrame""")

    def to_cpu(self):
        raise NotImplementedError(""" Move data to CPU memory """)

    def to_gpu(self):
        raise NotImplementedError(""" Move data to GPU memory """)

    @property
    def _path_partition_map(self):
        return None

    @property
    def num_rows(self):
        raise NotImplementedError(""" Returns the number of rows in the dataset """)

    def validate_dataset(self, **kwargs):
        raise NotImplementedError(""" Returns True if the raw data is efficient for NVTabular """)

    @classmethod
    def regenerate_dataset(cls, dataset, output_path, columns=None, **kwargs):
        raise NotImplementedError(""" Regenerate a dataset with optimal properties """)

    def sample_data(self, n=1):
        """Return a sample of real data from the dataset

        Sample the partitions of the underlying Dask collection
        until a non-empty partition is found. Then, use the first
        ``n`` rows of that partition to infer dtype info. If no
        non-empty partitions are found, use the Dask metadata.
        """
        _ddf = self.to_ddf()
        for partition_index in range(_ddf.npartitions):
            _head = _ddf.partitions[partition_index].head(n)
            if len(_head):
                return _head
        return _ddf._meta
