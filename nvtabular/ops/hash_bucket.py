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
from cudf.utils.dtypes import is_list_dtype
from nvtx import annotate

from .categorify import SetBuckets, _encode_list_column
from .operator import CAT
from .transform_operator import DFOperator


class HashBucket(DFOperator):
    """
    This op maps categorical columns to a contiguous integer range
    by first hashing the column then modulating by the number of
    buckets as indicated by `num_buckets`.

    Example usage::

        cat_names = ["feature_a", "feature_b"]
        cont_names = ...
        label_name = ...
        workflow = nvt.Workflow(
            cat_names=cat_names, cont_names=cont_names, label_name=label_names
        )

        # this will hash both features a and b to 100 buckets
        op = nvt.ops.HashBucket(100)

        # for different numbers of buckets per feature, initialize with a dict
        op = nvt.ops.HashBucket({"feature_a": 100, "feature_b": 50})

        # or, equivalently
        op = nvt.ops.HashBucket(
            num_buckets=[100, 50], columns=["feature_a", "feature_b"]
        )

        workflow.add_cat_preprocess(op)

    The output of this op would be::

        workflow.finalize()
        gdf = cudf.DataFrame({
            "feature_a": [101588, 2214177, 92855],
            "feature_b": ["foo", "bar", "baz"]
        })
        workflow.apply_ops(gdf)

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
    num_buckets : int, list of int, or dictionary:{column: num_hash_buckets}
        Column-wise modulo to apply after hash function. Note that this
        means that the corresponding value will be the categorical cardinality
        of the transformed categorical feature. If given as an int, that value
        will be used as the number of "hash buckets" for every feature. If
        a list is provided, it must be of the same length as `columns` (which
        should not be `None`), and the values will correspond to the number
        of buckets to use for the feature specified at the same index in
        `columns`. If a dictionary is passed, it will be used to specify
        explicit mappings from a column name to a number of buckets. In
        this case, only the columns specified in the keys of `num_buckets`
        will be transformed.
    columns: list of str or None
        Column names to apply hash bucket transformation to. Ignored if
        `num_buckets` is a `dict`. If `num_buckets` is given as a list,
        `columns` must not be None and have the same length. If left
        as None, transformation will be applied to all categorical
        columns. Note that this case is only used if `num_buckets` is
        an `int`.
    """

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
            if is_list_dtype(gdf[col].dtype):
                encoded = _encode_list_column(gdf[col], gdf[col].list.leaves.hash_values() % nb)
            else:
                encoded = gdf[col].hash_values() % nb

            new_gdf[new_col] = encoded
        return new_gdf

    @property
    def req_stats(self):
        return [
            SetBuckets(
                columns=self.columns,
                num_buckets=self.num_buckets,
            )
        ]
