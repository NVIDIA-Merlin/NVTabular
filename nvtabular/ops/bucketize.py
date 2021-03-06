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
import cudf
from nvtx import annotate
from six import callable

from .operator import ColumnNames, Operator


class Bucketize(Operator):
    """This operation transforms continuous features into categorical features
    with bins based on the provided bin boundaries.

    Example usage::

        #
        cont_names = ['cont1', 'cont2']
        boundaries = {
            'cont1': [-50, 0, 50],
            'cont2': [0, 25, 50, 75, 100]
        }
        bucketize_op = cont_names >> ops.Bucketize(boundaries)
        processor = nvt.Workflow(bucketize_op)

    Parameters
    ----------
    boundaries : int, dict or callable
        Defines how to transform the continous values into bins
    """

    def __init__(self, boundaries):
        # transform boundaries into a lookup function on column names
        if isinstance(boundaries, (list, tuple)):
            self.boundaries = lambda col: boundaries
        elif isinstance(boundaries, dict):
            self.boundaries = lambda col: boundaries[col]
        elif callable(boundaries):
            self.boundaries = boundaries
        else:
            raise TypeError(
                "`boundaries` must be dict, callable, or list, got type {}".format(type(boundaries))
            )
        super().__init__()

    @annotate("Bucketize_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, gdf: cudf.DataFrame) -> cudf.DataFrame:
        boundaries = {name: self.boundaries(name) for name in columns}
        new_gdf = cudf.DataFrame()
        for col, b in boundaries.items():
            # TODO: should just be using cupy.digitize but it's not in 7.8
            val = 0
            for boundary in b:
                val += (gdf[col] >= boundary).astype("int")
            new_gdf[col] = val
        return new_gdf

    transform.__doc__ = Operator.transform.__doc__
