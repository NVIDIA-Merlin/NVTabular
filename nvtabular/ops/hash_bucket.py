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
from typing import Dict, Union

import cudf
from cudf.utils.dtypes import is_list_dtype
from nvtx import annotate

from ..dispatch import _encode_list_column
from .categorify import _emb_sz_rule, _get_embedding_order
from .operator import ColumnNames, Operator


class HashBucket(Operator):
    """
    This op maps categorical columns to a contiguous integer range by first
    hashing the column, then reducing modulo the number of buckets.

    Example usage::

        cat_names = ["feature_a", "feature_b"]

        # this will hash both features a and b to 100 buckets
        hash_features = cat_names >> ops.HashBucket({"feature_a": 100, "feature_b": 50})
        processor = nvtabular.Workflow(hash_features)


    The output of this op would be::

           feature_a  feature_b
        0         90         11
        1         70         40
        2         52          9

    If you would like to do frequency capping or frequency hashing,
    you should use Categorify op instead. See
    `Categorify op <https://github.com/NVIDIA/NVTabular/blob/main/nvtabular/ops/categorify.py#L43>`_
    for example usage.


    Parameters
    ----------
    num_buckets : int or dictionary:{column: num_hash_buckets}
        Column-wise modulo to apply after hash function. Note that this
        means that the corresponding value will be the categorical cardinality
        of the transformed categorical feature. If given as an int, that value
        will be used as the number of "hash buckets" for every feature.
        If a dictionary is passed, it will be used to specify
        explicit mappings from a column name to a number of buckets. In
        this case, only the columns specified in the keys of `num_buckets`
        will be transformed.
    """

    def __init__(self, num_buckets: Union[int, Dict[str, int]]):
        if isinstance(num_buckets, dict):
            self.num_buckets = num_buckets
        elif isinstance(num_buckets, int):
            self.num_buckets = num_buckets
        else:
            raise TypeError(
                "`num_buckets` must be dict, iterable, or int, got type {}".format(
                    type(num_buckets)
                )
            )
        super(HashBucket, self).__init__()

    @annotate("HashBucket_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, gdf: cudf.DataFrame) -> cudf.DataFrame:
        if isinstance(self.num_buckets, int):
            num_buckets = {name: self.num_buckets for name in columns}
        else:
            num_buckets = self.num_buckets

        for col, nb in num_buckets.items():
            if is_list_dtype(gdf[col].dtype):
                gdf[col] = _encode_list_column(gdf[col], gdf[col].list.leaves.hash_values() % nb)
            else:
                gdf[col] = gdf[col].hash_values() % nb
        return gdf

    transform.__doc__ = Operator.transform.__doc__

    def get_embedding_sizes(self, columns):
        columns = _get_embedding_order(columns)
        if isinstance(self.num_buckets, int):
            embedding_size = _emb_sz_rule(self.num_buckets)
            return {col: embedding_size for col in columns}
        else:
            return {col: _emb_sz_rule(self.num_buckets[col]) for col in columns}
