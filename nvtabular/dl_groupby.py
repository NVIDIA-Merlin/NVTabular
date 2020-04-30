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
    stats : list of str, default ['count']
        count of groups = ['count']
        sum of cont_col = ['sum']
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
    ):
        if col is None:
            raise ValueError("cat_names cannot be None for group by operations.")

        if cont_col is None:
            if "count" not in stats:
                raise ValueError(
                    "count operations is only supported when there is no continuous columns."
                )

        supported_ops = ["count", "sum"]
        for ops in stats:
            if ops not in supported_ops:
                raise ValueError(ops + " operation is not supported.")

        self.col = col
        self.col_count = col_count
        self.cont_col = cont_col
        self.stats_names = stats
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.order_column_name = order_column_name
        self.means_host = []
        self.sums_host = []
        self.vars_host = []
        self.counts_host = []

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
                if "sum" in self.stats_names:
                    col_names.append(self.col + "_" + self.cont_col[i] + "_sum")

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

        # print(col_names)
        return stats_joined[col_names]

    def fit(self, gdf):
        """
        Calculates the requested group stats of gdf and
        stores results in the host memory.

        Parameters
        -----------
        gdf : cudf DataFrame

        """
        groups = gdf.groupby([self.col])

        if self.cont_col is not None:
            if "mean" in self.stats_names:
                means_part = groups.mean()
                means_part[self.col] = means_part.index
                self.means_host.append(means_part.to_pandas())

            if "sum" in self.stats_names:
                sums_part = groups.sum()
                sums_part[self.col] = sums_part.index
                self.sums_host.append(sums_part.to_pandas())

        if "count" in self.stats_names:
            counts_part = groups.count()
            counts_part[self.col] = counts_part.index
            self.counts_host.append(counts_part.to_pandas())

    def fit_finalize(self):
        """
        Finalizes the stats calculation.

        """

        self.stats = pd.DataFrame()

        if self.cont_col is not None:
            if "sum" in self.stats_names:
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

        if "count" in self.stats_names:
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

        self.stats[self.col] = self.stats.index
        self.stats.reset_index(drop=True, inplace=True)

        return self.stats.shape[0]
