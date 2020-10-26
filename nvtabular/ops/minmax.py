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


class MinMax(StatOperator):
    """
    MinMax operation calculates min and max statistics of features.

    Parameters
    -----------
    columns :
    batch_mins : list of float, default None
    batch_maxs : list of float, default None
    mins : list of float, default None
    maxs : list of float, default None
    """

    def __init__(self, columns=None, batch_mins=None, batch_maxs=None, mins=None, maxs=None):
        super().__init__(columns=columns)
        self.batch_mins = batch_mins if batch_mins is not None else {}
        self.batch_maxs = batch_maxs if batch_maxs is not None else {}
        self.mins = mins if mins is not None else {}
        self.maxs = maxs if maxs is not None else {}

    @annotate("MinMax_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        dask_stats = {}
        dask_stats["mins"] = ddf[cols].min()
        dask_stats["maxs"] = ddf[cols].max()
        return dask_stats

    @annotate("MinMax_finalize", color="green", domain="nvt_python")
    def finalize(self, stats):
        for col in stats["mins"].index.values_host:
            self.mins[col] = stats["mins"][col]
            self.maxs[col] = stats["maxs"][col]

    def registered_stats(self):
        return ["mins", "maxs", "batch_mins", "batch_maxs"]

    def stats_collected(self):
        result = [
            ("mins", self.mins),
            ("maxs", self.maxs),
            ("batch_mins", self.batch_mins),
            ("batch_maxs", self.batch_maxs),
        ]
        return result

    def clear(self):
        self.batch_mins = {}
        self.batch_maxs = {}
        self.mins = {}
        self.maxs = {}
        return
