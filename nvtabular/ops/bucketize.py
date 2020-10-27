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
from six import callable

from .operator import CAT, CONT
from .transform_operator import TransformOperator


class Bucketize(TransformOperator):
    """"""

    default_in = CONT
    default_out = CAT

    def __init__(self, boundaries, columns=None, **kwargs):
        if isinstance(boundaries, dict):
            columns = [i for i in boundaries.keys()]
            self.boundaries = boundaries
        elif callable(boundaries):
            self.boundaries = boundaries
        elif isinstance(boundaries, (tuple, list)):
            if any([isinstance(x, (tuple, list)) for x in boundaries]):
                assert all([isinstance(x, (tuple, list)) for x in boundaries])
                assert columns is not None
                assert len(columns) == len(boundaries)
                self.boundaries = {col: b for col, b in zip(columns, boundaries)}
            else:
                self.boundaries = lambda name: boundaries
        else:
            raise TypeError(
                "`boundaries` must be dict, callable, or iterable, got type {}".format(
                    type(boundaries)
                )
            )
        super().__init__(columns=columns, **kwargs)

    @annotate("HashedCross_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        cont_names = target_columns
        if callable(self.boundaries):
            boundaries = {name: self.boundaries(name) for name in cont_names}
        else:
            boundaries = self.boundaries

        new_gdf = cudf.DataFrame()
        for col, b in boundaries.items():
            # TODO: should just be using cupy.digitize but it's not in 7.8
            val = 0
            for boundary in b:
                val += (gdf[col] >= boundary).astype("int")
            new_col = f"{col}_{self._id}"
            new_gdf[new_col] = val
        return new_gdf
