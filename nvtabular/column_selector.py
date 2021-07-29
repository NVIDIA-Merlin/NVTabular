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
from dataclasses import dataclass
from typing import List, Union


@dataclass
class ColumnSelector:
    names: List[Union[str, List[str]]]

    def __getitem__(self, index):
        return self.names[index]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __add__(self, other):
        if isinstance(other, ColumnSelector):
            return ColumnSelector(self.names + other.names)
        else:
            return ColumnSelector(self.names + other)

    def __radd__(self, other):
        return self + other
