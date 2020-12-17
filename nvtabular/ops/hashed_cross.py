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

import cudf
from nvtx import annotate

from .operator import Operator


class HashedCross(Operator):
    def __init__(self, num_buckets):
        super().__init__()
        self.num_buckets = num_buckets

    @annotate("HashedCross_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns, gdf: cudf.DataFrame):
        new_gdf = cudf.DataFrame()
        val = 0
        for column in columns:
            val ^= gdf[column].hash_values()  # or however we want to do this aggregation
        # TODO: support different size buckets per cross
        val = val % self.num_buckets
        new_gdf["_X_".join(columns)] = val
        return new_gdf

    def output_column_names(self, columns):
        return ["_X_".join(columns)]
