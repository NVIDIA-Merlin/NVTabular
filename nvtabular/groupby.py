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
import cupy as cp
import pandas as pd
import rmm


class GroupByMomentsCal(object):
    """
    This is the class that GroupByMoments uses to
    calculate the basic statistics of the data that
    is grouped by a categorical feature.

    Parameters
    -----------
    col : str
        column name
    col_count : str
        column name to get group counts
    cont_col : list of str
        pre-calculated unique values.
    stats : list of str or set of str, default ['count']
        count of groups = ['count']
        sum of cont_col = ['sum']
        mean of cont_col = ['mean']
        var of cont_col = ['var']
        std of cont_col = ['std']
    limit_frac : float, default 0.1
        fraction of memory to use during unique id calculation.
    gpu_mem_util_limit : float, default 0.8
        GPU memory utilization limit during frequency based
        calculation. If limit is exceeded, unique ids are moved
        to host memory.
    gpu_mem_trans_use : float, default 0.8
        GPU memory utilization limit during transformation. How much
        GPU memory will be used during transformation is calculated
        using this parameter.
    order_column_name : str, default "order-nvtabular"
        a column name to be used to preserve the order of input data.
        cudf's merge function doesn't preserve the order of the data
        and this column name is used to create a column with integer
        values in ascending order.
    ddof : int, default "1"
        Delta Degrees of Freedom. The divisor used in calculations is
        N - ddof, where N represents the number of elements.
    """

    def __init__(
        self,
        col,
        col_count,
        cont_col,
        stats=["count"],
        limit_frac=0.1,
        gpu_mem_util_limit=0.8,
        gpu_mem_trans_use=0.8,
        order_column_name="order-nvtabular",
        ddof=1,
    ):
        if col is None:
            raise ValueError("cat_names cannot be None for group by operations.")

        if cont_col is None:
            if "count" not in stats:
                raise ValueError(
                    "count operations is only supported when there is no continuous columns."
                )

        self.supported_ops = ["count", "sum", "mean", "var", "std"]
        for ops in stats:
            if ops not in self.supported_ops:
                raise ValueError(ops + " operation is not supported.")

        self.col = col
        if isinstance(cont_col, str):
            cont_col = [cont_col]
        self.col_count = col_count
        self.cont_col = cont_col
        if isinstance(stats, list):
            stats = set(stats)
        self.stats_names = stats
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.order_column_name = order_column_name
        self.means_host = []
        self.sums_host = []
        self.vars_host = []
        self.counts_host = []
        self.ddof = ddof

    def merge(self, gdf):
        """
        Merges gdf with the calculated group stats.

        Parameters
        -----------
        gdf : cudf DataFrame

        Returns
        -----------
        stats_joined: cudf DataFrame
        """

        order = cudf.Series(cp.arange(gdf.shape[0]))
        gdf[self.order_column_name] = order

        col_names = []

        if self.cont_col is not None:
            for i in range(len(self.cont_col)):
                col_prefix = f"{self.col}_{self.cont_col[i]}_"
                col_names.extend(col_prefix + stat for stat in self.stats_names if stat != "count")

        if "count" in self.stats_names:
            col_names.append(self.col + "_count")

        avail_gpu_mem = rmm.get_info().free
        sub_stats_size = int(avail_gpu_mem * self.gpu_mem_trans_use / (self.stats.shape[1] * 8))
        if sub_stats_size == 0:
            sub_stats_size = 1

        stats_joined = None
        i = 0
        while i < self.stats.shape[0]:
            sub_stats = cudf.from_pandas(self.stats.iloc[i : i + sub_stats_size])
            joined = gdf[[self.col, self.order_column_name]].merge(
                sub_stats, on=[self.col], how="left"
            )
            joined = joined.sort_values(self.order_column_name)
            joined.reset_index(drop=True, inplace=True)

            if stats_joined is None:
                stats_joined = joined[col_names].copy()
            else:
                stats_joined = stats_joined.add(joined[col_names], fill_value=0)

            i = i + sub_stats_size

        joined = cudf.Series([])
        gdf.drop(columns=[self.order_column_name], inplace=True)

        return stats_joined[col_names]

    def fit(self, gdf):
        """
        Calculates the requested group stats of gdf and
        stores results in the host memory.

        Parameters
        -----------
        gdf : cudf DataFrame

        """
        if self.cont_col is None:
            groups = gdf[[self.col] + [self.col_count]].groupby([self.col])
        else:
            groups = gdf[[self.col] + self.cont_col + [self.col_count]].groupby([self.col])

        if self.cont_col is not None:
            if self._el_in_stats_names({"sum", "mean", "std", "var"}):
                sums_part = groups.sum()
                self.sums_host.append(sums_part.to_pandas())
            if self._el_in_stats_names({"std", "var"}):
                var_part = groups.std(ddof=self.ddof) ** 2
                self.vars_host.append(var_part.to_pandas())

        if self._el_in_stats_names({"count", "mean", "std", "var"}):
            counts_part = groups.count()
            self.counts_host.append(counts_part.to_pandas())

    def _el_in_stats_names(self, elements):
        return not self.stats_names.isdisjoint(elements)

    def fit_finalize(self):
        """
        Finalizes the stats calculation.

        """

        self.stats = pd.DataFrame()

        if "count" in self.stats_names and not (self._el_in_stats_names({"mean", "std", "var"})):
            counts_dev = cudf.DataFrame([])
            for i in range(len(self.counts_host)):
                counts_part = cudf.from_pandas(self.counts_host.pop())
                if counts_dev.shape[0] == 0:
                    counts_dev = counts_part
                else:
                    counts_dev = counts_dev.add(counts_part, fill_value=0)

            self.counts = counts_dev.to_pandas()
            new_col = self.col + "_count"
            self.stats[new_col] = self.counts[self.col_count]

        if self.cont_col is not None:
            if "sum" in self.stats_names and not (self._el_in_stats_names({"mean", "std", "var"})):
                sums_dev = cudf.DataFrame([])
                for i in range(len(self.sums_host)):
                    sums_part = cudf.from_pandas(self.sums_host.pop())
                    if sums_dev.shape[0] == 0:
                        sums_dev = sums_part
                    else:
                        sums_dev = sums_dev.add(sums_part, fill_value=0)

                self.sums = sums_dev.to_pandas()
                for cont_name in self.cont_col:
                    new_col = self.col + "_" + cont_name + "_sum"
                    self.stats[new_col] = self.sums[cont_name]

            if self._el_in_stats_names({"mean", "std", "var"}):
                sums_dev = cudf.DataFrame([])
                counts_dev = cudf.DataFrame([])
                if self._el_in_stats_names({"std", "var"}):
                    var_dev = cudf.DataFrame([])
                for i in range(len(self.sums_host)):
                    sums_part = cudf.from_pandas(self.sums_host.pop())
                    counts_part = cudf.from_pandas(self.counts_host.pop())
                    if self._el_in_stats_names({"std", "var"}):
                        var_part = cudf.from_pandas(self.vars_host.pop())
                    if i == 0:
                        counts_dev = counts_part
                        sums_dev = sums_part
                        if self._el_in_stats_names({"std", "var"}):
                            var_dev = var_part
                    else:
                        if self._el_in_stats_names({"std", "var"}):
                            # n1*v1
                            var_dev = var_dev.mul(counts_dev)
                            # n2*v2
                            var_dev = var_dev.add(var_part.mul(counts_part), fill_value=0)
                            # n1*(m1-m12)**2
                            m12_tmp = sums_dev.add(sums_part, fill_value=0)
                            m12_tmp = m12_tmp.mul(1 / (counts_dev.add(counts_part, fill_value=0)))
                            var_dev = var_dev.add(
                                counts_dev.mul(
                                    ((sums_dev.mul(1 / counts_dev)).add(-1 * m12_tmp, fill_value=0))
                                    ** 2
                                ),
                                fill_value=0,
                            )
                            var_dev = var_dev.add(
                                counts_part.mul(
                                    (sums_part.mul(1 / counts_part).add(-1 * m12_tmp, fill_value=0))
                                    ** 2
                                ),
                                fill_value=0,
                            )
                            del m12_tmp

                        counts_dev = counts_dev.add(counts_part, fill_value=0)
                        sums_dev = sums_dev.add(sums_part, fill_value=0)
                        if self._el_in_stats_names({"std", "var"}):
                            var_dev = var_dev.mul(1 / counts_dev)

                result_map = {}
                if "count" in self.stats_names:
                    self.counts = counts_dev.to_pandas()
                    result_map["count"] = self.counts
                if "sum" in self.stats_names:
                    self.sums = sums_dev.to_pandas()
                    result_map["sum"] = self.sums
                if "mean" in self.stats_names:
                    mean_dev = sums_dev.mul(1 / counts_dev)
                    self.mean = mean_dev.to_pandas()
                    result_map["mean"] = self.mean
                if "var" in self.stats_names:
                    self.var = var_dev.to_pandas()
                    result_map["var"] = self.var
                if "std" in self.stats_names:
                    self.std = var_dev.sqrt().to_pandas()
                    result_map["std"] = self.std
                for cont_name in self.cont_col:
                    for su_op in self.supported_ops:
                        if su_op in self.stats_names:
                            if su_op == "count":
                                new_col = self.col + "_count"
                                self.stats[new_col] = result_map[su_op][cont_name]
                            else:
                                new_col = self.col + "_" + cont_name + "_" + su_op
                                self.stats[new_col] = result_map[su_op][cont_name]

        self.stats[self.col] = self.stats.index
        self.stats.reset_index(drop=True, inplace=True)

        return self.stats.shape[0]
