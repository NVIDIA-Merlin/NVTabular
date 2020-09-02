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
from cudf._lib.nvtx import annotate

from .stat_operator import StatOperator


class Moments(StatOperator):
    """
    Moments operation calculates some of the statistics of features including
    mean, variance, standarded deviation, and count.

    Parameters
    -----------
    columns :
    counts : list of float, default None
    means : list of float, default None
    varis : list of float, default None
    stds : list of float, default None
    """

    def __init__(self, columns=None, counts=None, means=None, varis=None, stds=None):
        super().__init__(columns=columns)
        self.counts = counts if counts is not None else {}
        self.means = means if means is not None else {}
        self.varis = varis if varis is not None else {}
        self.stds = stds if stds is not None else {}

    @annotate("Moments_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        dask_stats = {}
        dask_stats["count"] = ddf[cols].count()
        dask_stats["mean"] = ddf[cols].mean()
        dask_stats["std"] = ddf[cols].std()
        return dask_stats

    @annotate("Moments_finalize", color="green", domain="nvt_python")
    def finalize(self, dask_stats):
        for col in dask_stats["count"].index.values_host:
            self.counts[col] = float(dask_stats["count"][col])
            self.means[col] = float(dask_stats["mean"][col])
            self.stds[col] = float(dask_stats["std"][col])
            self.varis[col] = float(self.stds[col] * self.stds[col])

    def registered_stats(self):
        return ["means", "stds", "vars", "counts"]

    def stats_collected(self):
        result = [
            ("means", self.means),
            ("stds", self.stds),
            ("vars", self.varis),
            ("counts", self.counts),
        ]
        return result

    def clear(self):
        self.counts = {}
        self.means = {}
        self.varis = {}
        self.stds = {}
        return
