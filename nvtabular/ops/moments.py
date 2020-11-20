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
import math

import cudf
import numpy as np
import pandas as pd
from dask.base import tokenize
from dask.dataframe.core import _concat
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from nvtx import annotate

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
        return _custom_moments(ddf[cols])

    @annotate("Moments_finalize", color="green", domain="nvt_python")
    def finalize(self, dask_stats):
        for col in dask_stats.index:
            self.counts[col] = float(dask_stats["count"].loc[col])
            self.means[col] = float(dask_stats["mean"].loc[col])
            self.stds[col] = float(dask_stats["std"].loc[col])
            self.varis[col] = float(dask_stats["var"].loc[col])

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


def _custom_moments(ddf, split_every=32):

    # Build custom task graph to gather stat moments
    dsk = {}
    token = tokenize(ddf)
    tree_reduce_name = "chunkwise-moments-" + token
    result_name = "global-moments-" + token
    for p in range(ddf.npartitions):
        # Gather necessary statstics on each partition.
        dsk[(tree_reduce_name, p, 0)] = (_chunkwise_moments, (ddf._name, p))

    # Build reduction tree
    parts = ddf.npartitions
    widths = [parts]
    while parts > 1:
        parts = math.ceil(parts / split_every)
        widths.append(parts)
    height = len(widths)
    for depth in range(1, height):
        for group in range(widths[depth]):

            p_max = widths[depth - 1]
            lstart = split_every * group
            lstop = min(lstart + split_every, p_max)
            node_list = [(tree_reduce_name, p, depth - 1) for p in range(lstart, lstop)]

            dsk[(tree_reduce_name, group, depth)] = (
                _tree_node_moments,
                node_list,
            )

    dsk[result_name] = (_finalize_moments, (tree_reduce_name, 0, height - 1))

    graph = HighLevelGraph.from_collections(result_name, dsk, dependencies=[ddf])

    return Delayed(result_name, graph)


def _chunkwise_moments(df):
    df2 = cudf.DataFrame()
    for col in df.columns:
        df2[col] = df[col].astype("float64").pow(2)
    vals = {
        "df-count": df.count().to_frame().transpose(),
        "df-sum": df.sum().astype("float64").to_frame().transpose(),
        "df2-sum": df2.sum().to_frame().transpose(),
    }
    # NOTE: Perhaps we should convert to pandas here
    # (since we know the results should be small)?
    del df2
    return vals


def _tree_node_moments(inputs):
    out = {}
    for val in ["df-count", "df-sum", "df2-sum"]:
        df_list = [x.get(val, None) for x in inputs]
        df_list = [df for df in df_list if df is not None]
        out[val] = _concat(df_list, ignore_index=True).sum().to_frame().transpose()
    return out


def _finalize_moments(inp, ddof=1):
    n = inp["df-count"].iloc[0].to_pandas()
    x = inp["df-sum"].iloc[0].to_pandas()
    x2 = inp["df2-sum"].iloc[0].to_pandas()

    # Use sum-squared approach to get variance
    var = x2 - x ** 2 / n
    div = n - ddof
    div[div < 1] = 1  # Avoid division by 0
    var /= div

    # Set appropriate NaN elements
    # (since we avoided 0-division)
    var[(n - ddof) == 0] = np.nan

    # Construct output DataFrame
    out = pd.DataFrame(index=inp["df-count"].columns)
    out["count"] = n
    out["sum"] = x
    out["sum2"] = x2
    out["mean"] = x / n
    out["var"] = var
    out["std"] = np.sqrt(var)
    return out
