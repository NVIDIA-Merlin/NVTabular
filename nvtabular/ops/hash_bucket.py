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
import cudf
from cudf._lib.nvtx import annotate

from .operator import CAT
from .transform_operator import TransformOperator


class HashBucket(TransformOperator):
    default_in = CAT
    default_out = CAT

    def __init__(self, num_buckets, columns=None, **kwargs):
        if isinstance(num_buckets, dict):
            columns = [i for i in num_buckets.keys()]
            self.num_buckets = num_buckets
        elif isinstance(num_buckets, (tuple, list)):
            assert columns is not None
            assert len(columns) == len(num_buckets)
            self.num_buckets = {col: nb for col, nb in zip(columns, num_buckets)}
        elif isinstance(num_buckets, int):
            self.num_buckets = num_buckets
        else:
            raise TypeError(
                "`num_buckets` must be dict, iterable, or int, got type {}".format(
                    type(num_buckets)
                )
            )
        super(HashBucket, self).__init__(columns=columns, **kwargs)

    @annotate("HashBucket_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cat_names = target_columns
        if isinstance(self.num_buckets, int):
            num_buckets = {name: self.num_buckets for name in cat_names}
        else:
            num_buckets = self.num_buckets

        new_gdf = cudf.DataFrame()
        for col, nb in num_buckets.items():
            new_col = f"{col}_{self._id}"
            new_gdf[new_col] = gdf[col].hash_values() % nb
        return new_gdf
