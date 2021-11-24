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


class DataFrameIter:
    def __init__(self, ddf, columns=None, indices=None, partition_lens=None, epochs=1):
        self.indices = indices if isinstance(indices, list) else range(ddf.npartitions)
        self._ddf = ddf
        self.columns = columns
        self.partition_lens = partition_lens
        self.epochs = epochs

    def __len__(self):
        if self.partition_lens:
            # Use metadata-based partition-size information
            # if/when it is available.  Note that this metadata
            # will not be correct if rows where added or dropped
            # after IO (within Ops).
            return sum(self.partition_lens[i] for i in self.indices) * self.epochs
        if len(self.indices) < self._ddf.npartitions:
            return len(self._ddf.partitions[self.indices]) * self.epochs
        return len(self._ddf) * self.epochs

    def __iter__(self):
        for epoch in range(self.epochs):
            for i in self.indices:
                part = self._ddf.get_partition(i)
                if self.columns:
                    yield part[self.columns].compute(scheduler="synchronous")
                else:
                    yield part.compute(scheduler="synchronous")
        part = None
