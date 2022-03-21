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
import math

import numpy as np
import pandas as pd
from dask.base import tokenize
from dask.dataframe.core import _concat
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph

from merlin.core.dispatch import flatten_list_column_values, is_list_dtype


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
    vals = {name: type(df)() for name in ["count", "sum", "squaredsum"]}
    for name in df.columns:
        column = df[name]
        if is_list_dtype(column):
            column = flatten_list_column_values(column)

        vals["count"][name] = [column.count()]
        vals["sum"][name] = [column.sum().astype("float64")]
        vals["squaredsum"][name] = [column.astype("float64").pow(2).sum()]

    # NOTE: Perhaps we should convert to pandas here
    # (since we know the results should be small)?
    return vals


def _tree_node_moments(inputs):
    out = {}
    for val in ["count", "sum", "squaredsum"]:
        df_list = [x.get(val, None) for x in inputs]
        df_list = [df for df in df_list if df is not None]
        out[val] = _concat(df_list, ignore_index=True).sum().to_frame().transpose()
    return out


def _finalize_moments(inp, ddof=1):
    n = inp["count"].iloc[0]
    x = inp["sum"].iloc[0]
    x2 = inp["squaredsum"].iloc[0]
    if hasattr(n, "to_pandas"):
        n = n.to_pandas()
        x = x.to_pandas()
        x2 = x2.to_pandas()

    # Use sum-squared approach to get variance
    var = x2 - x ** 2 / n
    div = n - ddof
    div[div < 1] = 1  # Avoid division by 0
    var /= div

    # Set appropriate NaN elements
    # (since we avoided 0-division)
    var[(n - ddof) == 0] = np.nan

    # Construct output DataFrame
    out = pd.DataFrame(index=inp["count"].columns)
    out["count"] = n
    out["sum"] = x
    out["sum2"] = x2
    out["mean"] = x / n
    out["var"] = var
    out["std"] = np.sqrt(var)
    return out
