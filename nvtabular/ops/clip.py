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

from merlin.core.dispatch import DataFrameType, annotate

from .operator import ColumnSelector, Operator


class Clip(Operator):
    """
    This operation clips continuous values so that they are within a min/max bound.
    For instance by setting the min value to 0, you can replace all negative values with 0.
    This is helpful in cases where you want to log normalize values::

        # clip all continuous columns to be positive only, and then take the log of the clipped
        # columns
        columns = ColumnSelector(CONT_NAMES) >> Clip(min_value=0) >> LogOp()

    Parameters
    ----------
    min_value : float, default None
        The minimum value to clip values to: values less than this will be replaced with
        this value. Specifying ``None`` means don't apply a minimum threshold.
    max_value : float, default None
        The maximum value to clip values to: values greater than this will be replaced with
        this value. Specifying ``None`` means don't apply a maximum threshold.
    """

    def __init__(self, min_value=None, max_value=None):
        if min_value is None and max_value is None:
            raise ValueError("Must specify a min or max value to clip to")
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    @annotate("Clip_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        z_df = df[col_selector.names]
        if self.min_value is not None:
            z_df[z_df < self.min_value] = self.min_value
        if self.max_value is not None:
            z_df[z_df > self.max_value] = self.max_value
        return z_df

    transform.__doc__ = Operator.transform.__doc__
