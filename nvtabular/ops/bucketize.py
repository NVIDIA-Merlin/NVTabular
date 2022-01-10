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

import numpy as np
from packaging.version import Version

from nvtabular.dispatch import DataFrameType, _array, annotate
from nvtabular.graph.tags import Tags

from .operator import ColumnSelector, Operator


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
        Defines how to transform the continuous values into bins
    """

    def __init__(self, boundaries):
        # Check if we have cupy.digitize support
        try:
            import cupy

            self.use_digitize = Version(cupy.__version__) >= Version("8.0.0")
        except ImportError:
            # Assume cpu-backed data (since cupy is not even installed)
            self.use_digitize = True

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
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        boundaries = {name: self.boundaries(name) for name in col_selector.names}
        new_df = type(df)()
        for col, b in boundaries.items():
            if self.use_digitize:
                new_df[col] = np.digitize(
                    df[col].values,
                    _array(b, like_df=df),
                    right=False,
                )
            else:
                # TODO: Remove use_digitize=False code path
                # once cupy>=8.0.0 is required.
                val = 0
                for boundary in b:
                    val += (df[col] >= boundary).astype("int")
                new_df[col] = val
        return new_df

    def _compute_dtype(self, col_schema, input_schema):
        return col_schema.with_dtype(np.int64)

    def _compute_tags(self, col_schema, input_schema):
        col_schema = col_schema.with_tags([Tags.CATEGORICAL])
        return super()._compute_tags(col_schema, input_schema)

    transform.__doc__ = Operator.transform.__doc__
