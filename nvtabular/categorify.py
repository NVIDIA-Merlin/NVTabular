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


def _make_name(*args, sep="_"):
    return sep.join(args)


@annotate("top_level_groupby", color="green", domain="nvt_python")
def _top_level_groupby(
    gdf, cat_col_groups, tree_width, cont_cols, agg_list, on_host, concat_groups, name_sep
):
    sum_sq = "std" in agg_list or "var" in agg_list
    calculate_min = "min" in agg_list
    calculate_max = "max" in agg_list

    # Top-level operation for category-based groupby aggregations
    output = {}
    k = 0
    for i, cat_col_group in enumerate(cat_col_groups):

        if isinstance(cat_col_group, str):
            cat_col_group = [cat_col_group]
        cat_col_group_str = _make_name(*cat_col_group, sep=name_sep)

        if concat_groups and len(cat_col_group) > 1:
            # Concatenate columns and replace cat_col_group
            # with the single name
            df_gb = cudf.DataFrame()
            ignore_index = True
            df_gb[cat_col_group_str] = _concat([gdf[col] for col in cat_col_group], ignore_index)
            cat_col_group = [cat_col_group_str]
        else:
            # Compile aggregation dictionary and add "squared-sum"
            # column(s) (necessary when `cont_cols` is non-empty)
            df_gb = gdf[cat_col_group + cont_cols].copy(deep=False)

        agg_dict = {}
        agg_dict[cat_col_group[0]] = ["count"]
        for col in cont_cols:
            agg_dict[col] = ["sum"]
            if sum_sq:
                name = _make_name(col, "pow2", sep=name_sep)
                df_gb[name] = df_gb[col].pow(2)
                agg_dict[name] = ["sum"]

            if calculate_min:
                agg_dict[col].append("min")
            if calculate_max:
                agg_dict[col].append("max")

        # Perform groupby and flatten column index
        # (flattening provides better cudf support)
        gb = df_gb.groupby(cat_col_group, dropna=False).agg(agg_dict)
        gb.columns = [
            _make_name(*(tuple(cat_col_group) + name[1:]), sep=name_sep)
            if name[0] == cat_col_group[0]
            else _make_name(*(tuple(cat_col_group) + name), sep=name_sep)
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
def _mid_level_groupby(
    dfs, col_group, cont_cols, agg_list, freq_limit, on_host, concat_groups, name_sep
):

    if isinstance(col_group, str):
        col_group = [col_group]

    if concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=name_sep)]

    df = _concat(dfs, ignore_index=True)
    if on_host:
        df.reset_index(drop=True, inplace=True)
        df = cudf.from_pandas(df)
    groups = df.groupby(col_group, dropna=False)
    gb = groups.agg({col: _get_aggregation_type(col) for col in df.columns if col not in col_group})
    gb.reset_index(drop=False, inplace=True)

    name_count = _make_name(*(col_group + ["count"]), sep=name_sep)
    if freq_limit:
        gb = gb[gb[name_count] >= freq_limit]

    required = col_group.copy()
    if "count" in agg_list:
        required.append(name_count)

    ddof = 1
    for cont_col in cont_cols:
        name_sum = _make_name(*(col_group + [cont_col, "sum"]), sep=name_sep)
        if "sum" in agg_list:
            required.append(name_sum)

        if "mean" in agg_list:
            name_mean = _make_name(*(col_group + [cont_col, "mean"]), sep=name_sep)
            required.append(name_mean)
            gb[name_mean] = gb[name_sum] / gb[name_count]

        if "min" in agg_list:
            name_min = _make_name(*(col_group + [cont_col, "min"]), sep=name_sep)
            required.append(name_min)

        if "max" in agg_list:
            name_max = _make_name(*(col_group + [cont_col, "max"]), sep=name_sep)
            required.append(name_max)

        if "var" in agg_list or "std" in agg_list:
            n = gb[name_count]
            x = gb[name_sum]
            x2 = gb[_make_name(*(col_group + [cont_col, "pow2", "sum"]), sep=name_sep)]
            result = x2 - x ** 2 / n
            div = n - ddof
            div[div < 1] = 1
            result /= div
            result[(n - ddof) == 0] = np.nan

            if "var" in agg_list:
                name_var = _make_name(*(col_group + [cont_col, "var"]), sep=name_sep)
                required.append(name_var)
                gb[name_var] = result
            if "std" in agg_list:
                name_std = _make_name(*(col_group + [cont_col, "std"]), sep=name_sep)
                required.append(name_std)
                gb[name_std] = np.sqrt(result)

    if on_host:
        gb_pd = gb[required].to_pandas()
        del gb
        return gb_pd
    return gb[required]


def _get_aggregation_type(col):
    if col.endswith("_min"):
        return "min"
    elif col.endswith("_max"):
        return "max"
    else:
        return "sum"


@annotate("write_gb_stats", color="green", domain="nvt_python")
def _write_gb_stats(dfs, base_path, col_group, on_host, concat_groups, name_sep):
    if concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=name_sep)]
    ignore_index = True
    df = _concat(dfs, ignore_index)
    if on_host:
        df.reset_index(drop=True, inplace=True)
        df = cudf.from_pandas(df)
    if isinstance(col_group, str):
        col_group = [col_group]
    rel_path = "cat_stats.%s.parquet" % (_make_name(*col_group, sep=name_sep))
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
def _write_uniques(dfs, base_path, col_group, on_host, concat_groups, name_sep):
    if concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=name_sep)]
    ignore_index = True
    if isinstance(col_group, str):
        col_group = [col_group]
    df = _concat(dfs, ignore_index)
    if on_host:
        df.reset_index(drop=True, inplace=True)
        df = cudf.from_pandas(df)
    rel_path = "unique.%s.parquet" % (_make_name(*col_group, sep=name_sep))
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
    return {col: paths[i] for i, col in enumerate(cols)}


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
    concat_groups=False,
    name_sep="_",
):
    if not col_groups:
        return {}

    if concat_groups:
        if agg_list and agg_list != ["count"]:
            raise ValueError("Cannot use concat_groups=True with aggregations other than count")
        if agg_cols:
            raise ValueError("Cannot aggregate continuous-column stats with concat_groups=True")

    # Update tree_width
    tw = {}
    for col in col_groups:
        col = [col] if isinstance(col, str) else col
        col_str = _make_name(*col, sep=name_sep)
        if tree_width is None:
            tw[col_str] = 8
        elif isinstance(tree_width, int):
            tw[col_str] = tree_width
        else:
            tw[col_str] = tree_width.get(col_str, None) or 8
    tree_width = tw

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
            agg_list,
            on_host,
            concat_groups,
            name_sep,
        )
        k = 0
        for c, col in enumerate(col_groups):
            col = [col] if isinstance(col, str) else col
            col_str = _make_name(*col, sep=name_sep)
            for s in range(tree_width[col_str]):
                dsk[(split_name, p, c, s)] = (getitem, (level_1_name, p), k)
                k += 1

    col_groups_str = []
    for c, col in enumerate(col_groups):
        col = [col] if isinstance(col, str) else col
        col_str = _make_name(*col, sep=name_sep)
        col_groups_str.append(col_str)
        freq_limit_val = None
        if freq_limit:
            freq_limit_val = freq_limit[col_str] if isinstance(freq_limit, dict) else freq_limit
        for s in range(tree_width[col_str]):
            dsk[(level_2_name, c, s)] = (
                _mid_level_groupby,
                [(split_name, p, c, s) for p in range(ddf.npartitions)],
                col,
                agg_cols,
                agg_list,
                freq_limit_val,
                on_host,
                concat_groups,
                name_sep,
            )

        dsk[(level_3_name, c)] = (
            write_func,
            [(level_2_name, c, s) for s in range(tree_width[col_str])],
            out_path,
            col,
            on_host,
            concat_groups,
            name_sep,
        )

    dsk[finalize_labels_name] = (
        _finish_labels,
        [(level_3_name, c) for c, col in enumerate(col_groups)],
        col_groups_str,
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
    concat_groups=False,
    name_sep="_",
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
            concat_groups=concat_groups,
            name_sep=name_sep,
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
        concat_groups=concat_groups,
        name_sep=name_sep,
    )


def _encode(name, storage_name, path, gdf, cat_cache, na_sentinel=-1, freq_threshold=0):
    value = None
    selection_l = name if isinstance(name, list) else [name]
    selection_r = name if isinstance(name, list) else [storage_name]
    if path:
        if cat_cache is not None:
            cat_cache = (
                cat_cache if isinstance(cat_cache, str) else cat_cache.get(storage_name, "disk")
            )
            if len(gdf):
                with get_worker_cache("cats") as cache:
                    value = fetch_table_data(
                        cache, path, columns=selection_r, cache=cat_cache, cats_only=True
                    )
        else:
            value = cudf.io.read_parquet(path, index=False, columns=selection_r)
            value.index.name = "labels"
            value.reset_index(drop=False, inplace=True)

    if value is None:
        value = cudf.DataFrame()
        for c in selection_r:
            typ = gdf[selection_l[0]].dtype if len(selection_l) == 1 else gdf[c].dtype
            value[c] = cudf.Series([None], dtype=typ)
        value.index.name = "labels"
        value.reset_index(drop=False, inplace=True)

    if freq_threshold > 0:
        codes = cudf.DataFrame({"order": cp.arange(len(gdf))})
        for c in selection_l:
            codes[c] = gdf[c].copy()
        codes = codes.merge(
            value, left_on=selection_l, right_on=selection_r, how="left"
        ).sort_values("order")["labels"]
        codes.fillna(na_sentinel, inplace=True)
        return codes.values
    else:
        # Use `searchsorted` if we are using a "full" encoding
        labels = value[selection_r].searchsorted(gdf[selection_l], side="left", na_position="first")
        labels[labels >= len(value[selection_r])] = na_sentinel
        return labels


def _read_groupby_stat_df(path, name, cat_cache):
    if cat_cache is not None:
        cat_cache = cat_cache if isinstance(cat_cache, str) else cat_cache.get(name, "disk")
        with get_worker_cache("stats") as cache:
            if cache:
                return fetch_table_data(cache, path, cache=cat_cache)
    return cudf.io.read_parquet(path, index=False)


def _get_multicolumn_names(column_groups, gdf_columns, name_sep):
    cat_names = []
    multi_col_group = {}
    for col_group in column_groups:
        if isinstance(col_group, list):
            name = _make_name(*col_group, sep=name_sep)
            if name not in cat_names:
                cat_names.append(name)
                # TODO: Perhaps we should check that all columns from the group
                #       are in gdf here?
                multi_col_group[name] = col_group
        elif col_group in gdf_columns:
            cat_names.append(col_group)
    return cat_names, multi_col_group
