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
from io import BytesIO
from operator import getitem

import cudf
import cupy as cp
import numpy as np
from cudf._lib.nvtx import annotate
from dask.base import tokenize
from dask.dataframe.core import _concat
from dask.highlevelgraph import HighLevelGraph
from fsspec.core import get_fs_token_paths

from nvtabular.worker import get_worker_cache


@annotate("fetch_data", color="green", domain="nvt_python")
def fetch_data(cat_cache, col, path, cache="disk", kind="stats"):
    table = cat_cache.get(col, None)
    if table and not isinstance(table, cudf.DataFrame):
        if kind == "stats":
            return cudf.io.read_parquet(table, index=False)
        df = cudf.io.read_parquet(table, index=False, columns=[col])
        df.index.name = "labels"
        df.reset_index(drop=False, inplace=True)
        return df

    if table is None:
        if cache in ("device", "disk"):
            table = cudf.io.read_parquet(
                path, index=False, columns=None if kind == "stats" else [col]
            )
        elif cache == "host":
            with open(path, "rb") as f:
                cat_cache[col] = BytesIO(f.read())
            table = cudf.io.read_parquet(
                cat_cache[col], index=False, columns=None if kind == "stats" else [col]
            )
        if kind == "cats":
            table.index.name = "labels"
            table.reset_index(drop=False, inplace=True)
        if cache == "device":
            cat_cache[col] = table.copy(deep=False)
    return table


def _make_name(*args):
    return "_".join(args)


@annotate("top_level_groupby", color="green", domain="nvt_python")
def _top_level_groupby(gdf, cat_col_groups, tree_width, cont_cols, sum_sq, on_host):
    # Top-level operation for category-based groupby aggregations
    output = {}
    k = 0
    for i, cat_col_group in enumerate(cat_col_groups):

        if isinstance(cat_col_group, str):
            cat_col_group_str = cat_col_group
            cat_col_group = [cat_col_group]
        else:
            cat_col_group_str = str(cat_col_group)

        # Compile aggregation dictionary and add "squared-sum"
        # column(s) (necessary when `cont_cols` is non-empty)
        df_gb = gdf[cat_col_group + cont_cols].copy(deep=False)
        agg_dict = {}
        agg_dict[cat_col_group[0]] = ["count"]
        for col in cont_cols:
            agg_dict[col] = ["sum"]
            if sum_sq:
                name = _make_name(col, "pow2")
                df_gb[name] = df_gb[col].pow(2)
                agg_dict[name] = ["sum"]

        # Perform groupby and flatten column index
        # (flattening provides better cudf support)
        gb = df_gb.groupby(cat_col_group, dropna=False).agg(agg_dict)
        gb.columns = [
            _make_name(*(tuple(cat_col_group) + name[1:]))
            if name[0] == cat_col_group[0]
            else _make_name(*(tuple(cat_col_group) + name))
            for name in gb.columns.to_flat_index()
        ]
        gb.reset_index(inplace=True, drop=False)
        del df_gb

        # Split the result by the hash value of the categorical column
        for j, split in enumerate(
            gb.partition_by_hash(cat_col_group, tree_width[cat_col_group_str], keep_index=False)
        ):
            if on_host:
                output[k] = split.to_pandas()
            else:
                output[k] = split
            k += 1
        del gb
    return output


@annotate("mid_level_groupby", color="green", domain="nvt_python")
def _mid_level_groupby(dfs, col_group, cont_cols, agg_list, freq_limit, on_host):

    if isinstance(col_group, str):
        col_group = [col_group]

    ignore_index = True
    if on_host:
        gb = cudf.from_pandas(_concat(dfs, ignore_index)).groupby(col_group, dropna=False).sum()
    else:
        gb = _concat(dfs, ignore_index).groupby(col_group, dropna=False).sum()
    gb.reset_index(drop=False, inplace=True)

    name_count = _make_name(*(col_group + ["count"]))
    if freq_limit:
        gb = gb[gb[name_count] >= freq_limit]

    required = col_group.copy()
    if "count" in agg_list:
        required.append(name_count)

    ddof = 1
    for cont_col in cont_cols:
        name_sum = _make_name(*(col_group + [cont_col, "sum"]))
        if "sum" in agg_list:
            required.append(name_sum)

        if "mean" in agg_list:
            name_mean = _make_name(*(col_group + [cont_col, "mean"]))
            required.append(name_mean)
            gb[name_mean] = gb[name_sum] / gb[name_count]

        if "var" in agg_list or "std" in agg_list:
            n = gb[name_count]
            x = gb[name_sum]
            x2 = gb[_make_name(*(col_group + [cont_col, "pow2", "sum"]))]
            result = x2 - x ** 2 / n
            div = n - ddof
            div[div < 1] = 1
            result /= div
            result[(n - ddof) == 0] = np.nan

            if "var" in agg_list:
                name_var = _make_name(*(col_group + [cont_col, "var"]))
                required.append(name_var)
                gb[name_var] = result
            if "std" in agg_list:
                name_std = _make_name(*(col_group + [cont_col, "std"]))
                required.append(name_std)
                gb[name_std] = np.sqrt(result)

    if on_host:
        gb_pd = gb[required].to_pandas()
        del gb
        return gb_pd
    return gb[required]


@annotate("write_gb_stats", color="green", domain="nvt_python")
def _write_gb_stats(dfs, base_path, col_group, on_host):
    ignore_index = True
    df = _concat(dfs, ignore_index)
    if on_host:
        df = cudf.from_pandas(df)
    if isinstance(col_group, str):
        col_group = [col_group]
    rel_path = "cat_stats.%s.parquet" % (_make_name(*col_group))
    path = os.path.join(base_path, rel_path)
    if len(df):
        df = df.sort_values(col_group, na_position="first")
        df.to_parquet(path, write_index=False, compression=None)
    else:
        df_null = cudf.DataFrame({c: [None] for c in col_group})
        for c in col_group:
            df_null[c] = df_null[c].astype(df[c].dtype)
        df_null.to_parquet(path, write_index=False, compression=None)
    del df
    return path


@annotate("write_uniques", color="green", domain="nvt_python")
def _write_uniques(dfs, base_path, col_group, on_host):
    ignore_index = True
    if isinstance(col_group, str):
        col_group = [col_group]
    df = _concat(dfs, ignore_index)
    if on_host:
        df = cudf.from_pandas(df)
    rel_path = "unique.%s.parquet" % (_make_name(*col_group))
    path = "/".join([base_path, rel_path])
    if len(df):
        # Make sure first category is Null
        df = df.sort_values(col_group, na_position="first")
        new_cols = {}
        nulls_missing = False
        for col in col_group:
            if not df[col]._column.has_nulls:
                nulls_missing = True
                new_cols[col] = _concat(
                    [cudf.Series([None], dtype=df[col].dtype), df[col]], ignore_index
                )
            else:
                new_cols[col] = df[col].copy(deep=False)
        if nulls_missing:
            df = cudf.DataFrame(new_cols)
        df.to_parquet(path, write_index=False, compression=None)
    else:
        df_null = cudf.DataFrame({c: [None] for c in col_group})
        for c in col_group:
            df_null[c] = df_null[c].astype(df[c].dtype)
        df_null.to_parquet(path, write_index=False, compression=None)
    del df
    return path


def _finish_labels(paths, cols):
    return {str(col): paths[i] for i, col in enumerate(cols)}


def _groupby_to_disk(
    ddf,
    write_func,
    col_groups,
    agg_cols,
    agg_list,
    out_path,
    freq_limit,
    tree_width,
    on_host,
    stat_name="categories",
):
    if not col_groups:
        return {}

    # Update tree_width
    if tree_width is None:
        tree_width = {str(c): 8 for c in col_groups}
    elif isinstance(tree_width, int):
        tree_width = {str(c): tree_width for c in col_groups}
    else:
        for col in col_groups:
            if str(col) not in tree_width:
                tree_width[str(col)] = 8

    # Make dedicated output directory for the categories
    fs = get_fs_token_paths(out_path)[0]
    out_path = fs.sep.join([out_path, stat_name])
    fs.mkdirs(out_path, exist_ok=True)

    dsk = {}
    token = tokenize(ddf, col_groups, out_path, freq_limit, tree_width, on_host)
    level_1_name = "level_1-" + token
    split_name = "split-" + token
    level_2_name = "level_2-" + token
    level_3_name = "level_3-" + token
    finalize_labels_name = stat_name + "-" + token
    for p in range(ddf.npartitions):
        dsk[(level_1_name, p)] = (
            _top_level_groupby,
            (ddf._name, p),
            col_groups,
            tree_width,
            agg_cols,
            ("std" in agg_list or "var" in agg_list),
            on_host,
        )
        k = 0
        for c, col in enumerate(col_groups):
            for s in range(tree_width[str(col)]):
                dsk[(split_name, p, c, s)] = (getitem, (level_1_name, p), k)
                k += 1

    for c, col in enumerate(col_groups):
        for s in range(tree_width[str(col)]):
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
            [(level_2_name, c, s) for s in range(tree_width[str(col)])],
            out_path,
            col,
            on_host,
        )

    dsk[finalize_labels_name] = (
        _finish_labels,
        [(level_3_name, c) for c, col in enumerate(col_groups)],
        col_groups,
    )
    graph = HighLevelGraph.from_collections(finalize_labels_name, dsk, dependencies=[ddf])
    return graph, finalize_labels_name


def _category_stats(
    ddf,
    col_groups,
    agg_cols,
    agg_list,
    out_path,
    freq_limit,
    tree_width,
    on_host,
    stat_name="categories",
):
    # Check if we only need categories
    if agg_cols == [] and agg_list == []:
        agg_list = ["count"]
        return _groupby_to_disk(
            ddf,
            _write_uniques,
            col_groups,
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
        col_groups,
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
            if len(gdf):
                with get_worker_cache("cats") as cache:
                    value = fetch_data(cache, name, path, cache=cat_cache, kind="cats")
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
        with get_worker_cache("stats") as cache:
            if cache:
                return fetch_data(cache, name, path, cache=cat_cache, kind="stats")
    return cudf.io.read_parquet(path, index=False)
