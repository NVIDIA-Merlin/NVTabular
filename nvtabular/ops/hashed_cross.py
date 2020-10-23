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

from .operator import CAT
from .transform_operator import TransformOperator


class HashedCross(TransformOperator):
    """"""

    default_in = CAT
    default_out = CAT

    def __init__(self, crosses, num_buckets=None, **kwargs):
        if isinstance(crosses, dict):
            cross_sets = list(crosses.keys())
            num_buckets = [crosses[c] for c in cross_sets]
            crosses = cross_sets
        else:
            if num_buckets is None:
                raise ValueError("Must provide `num_buckets` if crosses is not dict")
            assert len(num_buckets) == len(crosses)
        assert all([isinstance(c, (tuple, list)) for c in crosses])
        assert all([len(c) > 1 for c in crosses])

        kwargs["replace"] = False
        kwargs["columns"] = list(set([i for j in crosses for i in j]))
        super().__init__(**kwargs)

        self.crosses = crosses
        self.num_buckets = num_buckets

    @annotate("HashedCross_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        new_gdf = cudf.DataFrame()
        for columns, bucket_size in zip(self.crosses, self.num_buckets):
            val = 0
            for column in columns:
                val ^= gdf[column].hash_values()  # or however we want to do this aggregation
            val = cudf.Series(val).hash_values() % bucket_size
            new_gdf["_X_".join(columns)] = val
        return new_gdf
