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
from dataclasses import dataclass, field
from typing import List


@dataclass
class ColumnSelector:
    _names: List[str]
    subgroups: List["ColumnSelector"] = field(default_factory=list)

    @property
    def names(self):
        names = []
        names += self._names
        for subgroup in self.subgroups:
            names += subgroup.names
        return names

    @property
    def grouped_names(self):
        names = []
        names += self._names
        for subgroup in self.subgroups:
            names.append(tuple(subgroup.names))
        return names

    def __post_init__(self):
        plain_names = []
        for name in self._names:
            if isinstance(name, str):
                plain_names.append(name)
            else:
                self.subgroups.append(ColumnSelector(name))
        self._names = plain_names

    def __getitem__(self, index):
        return self._names[index]

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __add__(self, other):
        if isinstance(other, ColumnSelector):
            return ColumnSelector(self._names + other._names, self.subgroups + other.subgroups)
        else:
            return ColumnSelector(self._names + other, self.subgroups)

    def __radd__(self, other):
        return self + other
