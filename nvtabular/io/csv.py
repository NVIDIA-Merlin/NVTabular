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
import dask_cudf

from .dataset_engine import DatasetEngine


class CSVDatasetEngine(DatasetEngine):
    """CSVDatasetEngine

    Thin wrapper around dask_cudf.read_csv.
    """

    def __init__(self, paths, part_size, storage_options=None, **kwargs):
        super().__init__(paths, part_size, storage_options)
        self._meta = {}
        self.csv_kwargs = kwargs
        self.csv_kwargs["storage_options"] = storage_options

        # CSV reader needs a list of files
        # (Assume flat directory structure if this is a dir)
        if len(self.paths) == 1 and self.fs.isdir(self.paths[0]):
            self.paths = self.fs.glob(self.fs.sep.join([self.paths[0], "*"]))

    def to_ddf(self, columns=None):
        if columns:
            return dask_cudf.read_csv(self.paths, chunksize=self.part_size, **self.csv_kwargs)[
                columns
            ]
        return dask_cudf.read_csv(self.paths, chunksize=self.part_size, **self.csv_kwargs)
