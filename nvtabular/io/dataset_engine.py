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

from dask.utils import natural_sort_key
from fsspec.core import get_fs_token_paths


class DatasetEngine:
    """Base class for Dask-powered IO engines. Engines must provide a ``to_ddf`` method."""

    def __init__(self, paths, part_size, storage_options=None):
        paths = sorted(paths, key=natural_sort_key)
        self.paths = paths
        self.part_size = part_size
        self.storage_options = storage_options
        fs, fs_token, _ = get_fs_token_paths(paths, mode="rb", storage_options=self.storage_options)
        self.fs = fs
        self.fs_token = fs_token

    def to_ddf(self, columns=None):
        raise NotImplementedError(""" Return a dask_cudf.DataFrame """)

    @property
    def num_rows(self):
        raise NotImplementedError(""" Returns the number of rows in the dataset """)
