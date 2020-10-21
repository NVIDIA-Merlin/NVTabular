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
from nvtx import annotate

from .stat_operator import StatOperator


class Median(StatOperator):
    """
    This operation calculates median of features.

    Parameters
    -----------
    columns :
    fill : float, default None
    batch_medians : list, default None
    medians : list, default None
    """

    def __init__(self, columns=None, fill=None, batch_medians=None, medians=None):
        super().__init__(columns=columns)
        self.fill = fill
        self.batch_medians = batch_medians if batch_medians is not None else {}
        self.medians = medians if medians is not None else {}

    @annotate("Median_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        # TODO: Use `method="tidigest"` when crick supports device
        dask_stats = ddf[cols].quantile(q=0.5, method="dask")
        return dask_stats

    @annotate("Median_finalize", color="green", domain="nvt_python")
    def finalize(self, dask_stats):
        for col in dask_stats.index.values_host:
            self.medians[col] = float(dask_stats[col])

    def registered_stats(self):
        return ["medians"]

    def stats_collected(self):
        result = [("medians", self.medians)]
        return result

    def clear(self):
        self.batch_medians = {}
        self.medians = {}
        return
