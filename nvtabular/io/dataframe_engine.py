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

from .dataset_engine import DatasetEngine


class DataFrameDatasetEngine(DatasetEngine):
    """DataFrameDatasetEngine allows NVT to interact with a dask_cudf.DataFrame object
    in the same way as a dataset on disk.
    """

    def __init__(self, ddf):
        self._ddf = ddf

    def to_ddf(self, columns=None):
        if isinstance(columns, list):
            return self._ddf[columns]
        elif isinstance(columns, str):
            return self._ddf[[columns]]
        return self._ddf

    @property
    def num_rows(self):
        return len(self._ddf)
