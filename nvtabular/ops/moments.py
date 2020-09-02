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
    subset : list
        Subset of statistic types to gather. Default is all supported
        types ("counts", "means", "stds", "vars")
    counts : list of float, default None
    means : list of float, default None
    varis : list of float, default None
    stds : list of float, default None
    """

    def __init__(self, columns=None, counts=None, means=None, varis=None, stds=None, subset=None):
        super().__init__(columns=columns)
        self.counts = counts if counts is not None else {}
        self.means = means if means is not None else {}
        self.varis = varis if varis is not None else {}
        self.stds = stds if stds is not None else {}
        _supported = ("means", "stds", "vars", "counts")
        self.subset = subset or _supported
        if len(self.subset) < 1:
            raise ValueError("Moments subset must include at least one Moment type")
        if not set(self.subset).issubset(set(_supported)):
            raise ValueError("One or more Moment types is not recognized.")

    @annotate("Moments_op", color="green", domain="nvt_python")
    def stat_logic(self, ddf, columns_ctx, input_cols, target_cols):
        cols = self.get_columns(columns_ctx, input_cols, target_cols)
        dask_stats = {}
        if "counts" in self.subset:
            dask_stats["count"] = ddf[cols].count()
        if "means" in self.subset:
            dask_stats["mean"] = ddf[cols].mean()
        if "stds" in self.subset or "vars" in self.subset:
            dask_stats["std"] = ddf[cols].std()
        return dask_stats

    @annotate("Moments_finalize", color="green", domain="nvt_python")
    def finalize(self, dask_stats):
        cols = dask_stats[self.subset[0][:-1]].index.values_host
        for col in cols:
            if "counts" in self.subset:
                self.counts[col] = float(dask_stats["count"][col])
            if "means" in self.subset:
                self.means[col] = float(dask_stats["mean"][col])
            if "stds" in self.subset or "var" in self.subset:
                self.stds[col] = float(dask_stats["std"][col])
                self.varis[col] = float(self.stds[col] * self.stds[col])

    def registered_stats(self):
        return self.subset

    def stats_collected(self):
        result = []
        if "means" in self.subset:
            result.append(("means", self.means))
        if "stds" in self.subset:
            result.append(("stds", self.stds))
        if "vars" in self.subset:
            result.append(("vars", self.varis))
        if "counts" in self.subset:
            result.append(("counts", self.counts))
        return result

    def clear(self):
        self.counts = {}
        self.means = {}
        self.varis = {}
        self.stds = {}
        return
