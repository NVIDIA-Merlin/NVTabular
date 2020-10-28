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
from nvtx import annotate

from .operator import CONT
from .transform_operator import TransformOperator


class DifferenceLag(TransformOperator):
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

    Once passed an appropiate dataset, this operator can be added to a nvtabular workflow to
    compute the lagged difference within a partition::

        # compute the delta in timestamp for each users session
        workflow.add_feature(DifferenceLag('userid', columns=["timestamp"]))

    Parameters
    -----------
    partition_cols : str or list of str
        Column or Columns that are used to partition the data.
    shift : int, default 1
        The number of rows to look backwards when computing the difference lag. Negative values
        indicate the number of rows to look forwards, making this compute the lead instead of lag.
    columns : list of str, default None
        Continuous columns to target for this op. If None, the operation will target all known
        continuous columns.
    replace: bool, default False
        Whether to replace existing columns or create new ones.
    """

    default_in = CONT
    default_out = CONT

    def __init__(self, partition_cols, shift=1, columns=None, replace=False):
        super(DifferenceLag, self).__init__(columns=columns, replace=replace)
        self.partition_cols = partition_cols
        self.shifts = [shift] if isinstance(shift, int) else shift

    @annotate("DifferenceLag_op", color="darkgreen", domain="nvt_python")
    def op_logic(self, gdf: cudf.DataFrame, target_columns: list, stats_context=None):
        # compute a mask indicating partition boundaries, handling multiple partition_cols
        # represent partition boundaries by None values
        output = {}
        for shift in self.shifts:
            mask = gdf[self.partition_cols] == gdf[self.partition_cols].shift(shift)
            if isinstance(mask, cudf.DataFrame):
                mask = mask.all(axis=1)
            mask[mask == False] = None  # noqa

            for col in target_columns:
                output[f"{col}_{self._id}_{shift}"] = (gdf[col] - gdf[col].shift(shift)) * mask
        return cudf.DataFrame(output)
