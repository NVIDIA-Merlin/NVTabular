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
from typing import List

import numpy as np


class TensorSeries:
    def __init__(self, name, values, offsets=None):
        self.name = name
        self.values = values
        self._offsets = offsets

    @property
    def offsets(self):
        """Return an array of integer values corresponding to
        indexes in values tensor (multihot) or integer
        representing length of values (singlehot)."""
        return self._offsets or len(self.values)

    @property
    def shape(self):
        """Return shape of tensors, whether a tuple (multihot)
        or single (singlehot)."""
        return self.offsets, 2 if self._offsets else 1

    def __getitem__(self, list_of_items):
        # should all be values
        list_of_items = list(list_of_items)
        # list of items should be indexes to grab
        if self._offsets:
            start_vals_idx = []
            end_vals_idx = []
            for index in list_of_items:
                start_vals_idx.append(self._offsets[index])
                end_vals_idx.append(self._offsets[index + 1])
            # grab values in lists
            values_in_lists = []
            for start, end in zip(start_vals_idx, end_vals_idx):
                values_in_lists += self.values[start:end]
            # recreate deltas from start and finish
            new_offsets = np.array(start_vals_idx) - np.array(end_vals_idx)
            return TensorSeries(self.name, values_in_lists, offsets=new_offsets)
        val = np.array(self.values)[list_of_items]
        return TensorSeries(self.name, val)

    def __repr__(self):
        results = {"name": self.name}
        results["values"] = self.values
        if self._offsets:
            results["offsets"] = self._offsets
        return str(results)

    def append(self, values, offsets=None):
        pass


class TensorDataFrame:
    def __init__(self, tensors: List[TensorSeries]):
        self.tensors = tensors

    def __getitem__(self, list_of_items):
        results = []
        for item in list_of_items:
            current_result = None
            for tensor in self.tensors:
                if tensor.name == item:
                    current_result = tensor
            if current_result:
                results.append(current_result)
            else:
                raise ValueError(f"KeyError {item} not found in tensors.")
        return TensorDataFrame(results)

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        for tensor in self.tensors:
            yield tensor

    def __repr__(self):
        dict_rep = {}
        for tensor in self.tensors:
            dict_rep[tensor.name] = tensor.values
        return str(dict_rep)

    def iloc(self):
        pass
