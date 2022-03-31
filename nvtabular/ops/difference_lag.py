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
import numpy

from merlin.core.dispatch import DataFrameType, annotate, is_dataframe_object
from merlin.schema import Tags

from .operator import ColumnSelector, Operator


class DifferenceLag(Operator):
    """Calculates the difference between two consecutive rows of the dataset. For instance, this
    operator can calculate the time since a user last had another interaction.

    This requires a dataset partitioned by one set of columns (userid) and sorted further by another
    set (userid, timestamp). The dataset must already be partitioned and sorted before being passed
    to the workflow. This can be easily done using dask-cudf::

        # get a nvt dataset and convert to a dask dataframe
        ddf = nvtabular.Dataset(PATHS).to_ddf()

        # partition the dask dataframe by userid, then sort by userid/timestamp
        ddf = ddf.shuffle("userid").sort_values(["userid", "timestamp"])

        # create a new nvtabular dataset on the partitioned/sorted values
        dataset = nvtabular.Dataset(ddf)

    Once passed an appropriate dataset, this operator can be used to create a workflow to
    compute the lagged difference within a partition::

        # compute the delta in timestamp for each users session
        diff_features = ["quantity"] >> ops.DifferenceLag(partition_cols=["userid"], shift=[1, -1])
        processor = nvtabular.Workflow(diff_features)

    Parameters
    -----------
    partition_cols : str or list of str
        Column or Columns that are used to partition the data.
    shift : int, default 1
        The number of rows to look backwards when computing the difference lag. Negative values
        indicate the number of rows to look forwards, making this compute the lead instead of lag.
    """

    def __init__(self, partition_cols, shift=1):
        super(DifferenceLag, self).__init__()

        if isinstance(partition_cols, str):
            partition_cols = [partition_cols]

        self.partition_cols = partition_cols
        self.shifts = [shift] if isinstance(shift, int) else shift

    @annotate("DifferenceLag_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        # compute a mask indicating partition boundaries, handling multiple partition_cols
        # represent partition boundaries by None values
        output = {}
        for shift in self.shifts:
            mask = df[self.partition_cols] == df[self.partition_cols].shift(shift)
            if is_dataframe_object(mask):
                mask = mask.fillna(False).all(axis=1)
            mask[mask == False] = None  # noqa pylint: disable=singleton-comparison

            for col in col_selector.names:
                name = self._column_name(col, shift)
                output[name] = (df[col] - df[col].shift(shift)) * mask
                output[name] = output[name].astype(self.output_dtype)
        return type(df)(output)

    transform.__doc__ = Operator.transform.__doc__

    @property
    def dependencies(self):
        return self.partition_cols

    def column_mapping(self, col_selector):
        column_mapping = {}
        for col in col_selector.names:
            for shift in self.shifts:
                output_col_name = self._column_name(col, shift)
                column_mapping[output_col_name] = [col]
        return column_mapping

    @property
    def output_tags(self):
        return [Tags.CONTINUOUS]

    @property
    def output_dtype(self):
        return numpy.float32

    def _column_name(self, col, shift):
        return f"{col}_difference_lag_{shift}"
