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

import os
from operator import getitem

import cudf
import cupy as cp
import numpy as np
from cudf._lib.nvtx import annotate
from dask.base import tokenize
from dask.dataframe.core import _concat
from dask.highlevelgraph import HighLevelGraph
from fsspec.core import get_fs_token_paths

from nvtabular.worker import fetch_table_data, get_worker_cache


def _make_name(*args):
    return "_".join(args)


@annotate("top_level_groupby", color="green", domain="nvt_python")
def _top_level_groupby(gdf, cat_cols, tree_width, cont_cols, sum_sq, on_host):
    # Top-level operation for category-based groupby aggregations
    output = {}
    k = 0
    for i, cat_col in enumerate(cat_cols):

        # Compile aggregation dictionary and add "squared-sum"
        # column(s) (necessary when `cont_cols` is non-empty)
        df_gb = gdf[[cat_col] + cont_cols].copy(deep=False)
        agg_dict = {}
        agg_dict[cat_col] = ["count"]
        for col in cont_cols:
            agg_dict[col] = ["sum"]
            if sum_sq:
                name = _make_name(col, "pow2")
                df_gb[name] = df_gb[col].pow(2)
                agg_dict[name] = ["sum"]

        # Perform groupby and flatten column index
        # (flattening provides better cudf support)
        gb = df_gb.groupby(cat_col, dropna=False).agg(agg_dict)
        gb.columns = [
            _make_name(*name) if name[0] == cat_col else _make_name(*((cat_col,) + name))
            for name in gb.columns.to_flat_index()
        ]
        gb.reset_index(inplace=True, drop=False)
        del df_gb

        # Split the result by the hash value of the categorical column
        for j, split in enumerate(
            gb.partition_by_hash([cat_col], tree_width[cat_col], keep_index=False)
        ):
            if on_host:
                output[k] = split.to_pandas()
            else:
                output[k] = split
            k += 1
        del gb
    return output


@annotate("mid_level_groupby", color="green", domain="nvt_python")
def _mid_level_groupby(dfs, col, cont_cols, agg_list, freq_limit, on_host):
    ignore_index = True
    if on_host:
        gb = cudf.from_pandas(_concat(dfs, ignore_index)).groupby(col, dropna=False).sum()
    else:
        gb = _concat(dfs, ignore_index).groupby(col, dropna=False).sum()
    gb.reset_index(drop=False, inplace=True)

    name_count = _make_name(col, "count")
    if freq_limit:
        gb = gb[gb[name_count] >= freq_limit]

    required = [col]
    if "count" in agg_list:
        required.append(name_count)

    ddof = 1
    for cont_col in cont_cols:
        name_sum = _make_name(col, cont_col, "sum")
        if "sum" in agg_list:
            required.append(name_sum)

        if "mean" in agg_list:
            name_mean = _make_name(col, cont_col, "mean")
            required.append(name_mean)
            gb[name_mean] = gb[name_sum] / gb[name_count]

        if "var" in agg_list or "std" in agg_list:
            n = gb[name_count]
            x = gb[name_sum]
            x2 = gb[_make_name(col, cont_col, "pow2", "sum")]
            result = x2 - x ** 2 / n
            div = n - ddof
            div[div < 1] = 1
            result /= div
            result[(n - ddof) == 0] = np.nan

            if "var" in agg_list:
                name_var = _make_name(col, cont_col, "var")
                required.append(name_var)
                gb[name_var] = result
            if "std" in agg_list:
                name_std = _make_name(col, cont_col, "std")
                required.append(name_std)
                gb[name_std] = np.sqrt(result)

    if on_host:
        gb_pd = gb[required].to_pandas()
        del gb
        return gb_pd
    return gb[required]


@annotate("write_gb_stats", color="green", domain="nvt_python")
def _write_gb_stats(dfs, base_path, col, on_host):
    ignore_index = True
    df = _concat(dfs, ignore_index)
    if on_host:
        df = cudf.from_pandas(df)
    rel_path = "cat_stats.%s.parquet" % (col)
    path = os.path.join(base_path, rel_path)
    if len(df):
        df = df.sort_values(col, na_position="first")
        df.to_parquet(path, write_index=False, compression=None)
    else:
        df_null = cudf.DataFrame({col: [None]})
        df_null[col] = df_null[col].astype(df[col].dtype)
        df_null.to_parquet(path, write_index=False, compression=None)
    del df
    return path


@annotate("write_uniques", color="green", domain="nvt_python")
def _write_uniques(dfs, base_path, col, on_host):
    ignore_index = True
    df = _concat(dfs, ignore_index)
    if on_host:
        df = cudf.from_pandas(df)
    rel_path = "unique.%s.parquet" % (col)
    path = "/".join([base_path, rel_path])
    if len(df):
        # Make sure first category is Null
        df = df.sort_values(col, na_position="first")
        if not df[col]._column.has_nulls:
            df = cudf.DataFrame(
                {col: _concat([cudf.Series([None], dtype=df[col].dtype), df[col]], ignore_index)}
            )
        df.to_parquet(path, write_index=False, compression=None)
    else:
        df_null = cudf.DataFrame({col: [None]})
        df_null[col] = df_null[col].astype(df[col].dtype)
        df_null.to_parquet(path, write_index=False, compression=None)
    del df
    return path


def _finish_labels(paths, cols):
    return {col: paths[i] for i, col in enumerate(cols)}


def _groupby_to_disk(
    ddf,
    write_func,
    cols,
    agg_cols,
    agg_list,
    out_path,
    freq_limit,
    tree_width,
    on_host,
    stat_name="categories",
):
    if not cols:
        return {}

    # Update tree_width
    if tree_width is None:
        tree_width = {c: 8 for c in cols}
    elif isinstance(tree_width, int):
        tree_width = {c: tree_width for c in cols}
    else:
        for col in cols:
            if col not in tree_width:
                tree_width[col] = 8

    # Make dedicated output directory for the categories
    fs = get_fs_token_paths(out_path)[0]
    out_path = fs.sep.join([out_path, stat_name])
    fs.mkdirs(out_path, exist_ok=True)

    dsk = {}
    token = tokenize(ddf, cols, out_path, freq_limit, tree_width, on_host)
    level_1_name = "level_1-" + token
    split_name = "split-" + token
    level_2_name = "level_2-" + token
    level_3_name = "level_3-" + token
    finalize_labels_name = stat_name + "-" + token
    for p in range(ddf.npartitions):
        dsk[(level_1_name, p)] = (
            _top_level_groupby,
            (ddf._name, p),
            cols,
            tree_width,
            agg_cols,
            ("std" in agg_list or "var" in agg_list),
            on_host,
        )
        k = 0
        for c, col in enumerate(cols):
            for s in range(tree_width[col]):
                dsk[(split_name, p, c, s)] = (getitem, (level_1_name, p), k)
                k += 1

    for c, col in enumerate(cols):
        for s in range(tree_width[col]):
            dsk[(level_2_name, c, s)] = (
                _mid_level_groupby,
                [(split_name, p, c, s) for p in range(ddf.npartitions)],
                col,
                agg_cols,
                agg_list,
                freq_limit,
                on_host,
            )

        dsk[(level_3_name, c)] = (
            write_func,
            [(level_2_name, c, s) for s in range(tree_width[col])],
            out_path,
            col,
            on_host,
        )

    dsk[finalize_labels_name] = (
        _finish_labels,
        [(level_3_name, c) for c, col in enumerate(cols)],
        cols,
    )
    graph = HighLevelGraph.from_collections(finalize_labels_name, dsk, dependencies=[ddf])
    return graph, finalize_labels_name


def _category_stats(
    ddf, cols, agg_cols, agg_list, out_path, freq_limit, tree_width, on_host, stat_name="categories"
):
    # Check if we only need categories
    if agg_cols == [] and agg_list == []:
        agg_list = ["count"]
        return _groupby_to_disk(
            ddf,
            _write_uniques,
            cols,
            agg_cols,
            agg_list,
            out_path,
            freq_limit,
            tree_width,
            on_host,
            stat_name=stat_name,
        )

    # Otherwise, getting category-statistics
    if isinstance(agg_cols, str):
        agg_cols = [agg_cols]
    if agg_list == []:
        agg_list = ["count"]
    return _groupby_to_disk(
        ddf,
        _write_gb_stats,
        cols,
        agg_cols,
        agg_list,
        out_path,
        freq_limit,
        tree_width,
        on_host,
        stat_name=stat_name,
    )


def _encode(name, path, gdf, cat_cache, na_sentinel=-1, freq_threshold=0):
    value = None
    if path:
        if cat_cache is not None:
            cat_cache = cat_cache if isinstance(cat_cache, str) else cat_cache.get(name, "disk")
            cache = get_worker_cache("cats") if len(gdf) else None
            if cache is not None:
                value = fetch_table_data(
                    cache, path, columns=[name], cache=cat_cache, cats_only=True
                )
        else:
            value = cudf.io.read_parquet(path, index=False, columns=[name])
            value.index.name = "labels"
            value.reset_index(drop=False, inplace=True)

    vals = gdf[name].copy(deep=False)
    if value is None:
        value = cudf.DataFrame({name: [None]})
        value[name] = value[name].astype(vals.dtype)
        value.index.name = "labels"
        value.reset_index(drop=False, inplace=True)

    if freq_threshold > 0:
        codes = cudf.DataFrame({name: vals.copy(), "order": cp.arange(len(vals))})
        codes = codes.merge(value, on=name, how="left").sort_values("order")["labels"]
        codes.fillna(na_sentinel, inplace=True)
        return codes.values
    else:
        # Use `searchsorted` if we are using a "full" encoding
        labels = value[name].searchsorted(vals, side="left", na_position="first")
        labels[labels >= len(value[name])] = na_sentinel
        return labels


def _read_groupby_stat_df(path, name, cat_cache):
    if cat_cache is not None:
        cat_cache = cat_cache if isinstance(cat_cache, str) else cat_cache.get(name, "disk")
        cache = get_worker_cache("stats")
        if cache:
            return fetch_table_data(cache, path, cache=cat_cache)
    return cudf.io.read_parquet(path, index=False)
