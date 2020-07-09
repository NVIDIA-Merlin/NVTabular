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
from dask.distributed import get_worker
from dask.highlevelgraph import HighLevelGraph
from fsspec.core import get_fs_token_paths

# Use global variable as the default
# cache when there are no distributed workers
DEFAULT_CACHE = None


class CategoryCache:
    def __init__(self):
        self.cat_cache = {"cats": {}, "stats": {}}

    @annotate("fetch_data", color="green", domain="nvt_python")
    def fetch_data(self, col, path, cache="disk", kind="stats"):
        table = self.cat_cache[kind].get(col, None)
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
                    self.cat_cache[kind][col] = BytesIO(f.read())
                table = cudf.io.read_parquet(
                    self.cat_cache[kind][col],
                    index=False,
                    columns=None if kind == "stats" else [col],
                )
            if kind == "cats":
                table.index.name = "labels"
                table.reset_index(drop=False, inplace=True)
            if cache == "device":
                self.cat_cache[kind][col] = table.copy(deep=False)
        return table


def _make_name(*args):
    return "_".join(args)


@annotate("top_level_groupby", color="green", domain="nvt_python")
def _top_level_groupby(gdf, cat_cols, split_out, cont_cols, sum_sq, on_host):
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
            gb.partition_by_hash([cat_col], split_out[cat_col], keep_index=False)
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
            df = cudf.DataFrame({col: _concat([cudf.Series([None]), df[col]], ignore_index)})
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
    ddf, write_func, cols, agg_cols, agg_list, out_path, freq_limit, split_out, on_host
):
    if not cols:
        return {}

    # Update split_out
    if split_out is None:
        split_out = {c: 1 for c in cols}
    elif isinstance(split_out, int):
        split_out = {c: split_out for c in cols}
    else:
        for col in cols:
            if col not in split_out:
                split_out[col] = 1

    # Make dedicated output directory for the categories
    fs = get_fs_token_paths(out_path)[0]
    out_path = fs.sep.join([out_path, "categories"])
    fs.mkdirs(out_path, exist_ok=True)

    dsk = {}
    token = tokenize(ddf, cols, out_path, freq_limit, split_out, on_host)
    level_1_name = "level_1-" + token
    split_name = "split-" + token
    level_2_name = "level_2-" + token
    level_3_name = "level_3-" + token
    finalize_labels_name = "categories-" + token
    for p in range(ddf.npartitions):
        dsk[(level_1_name, p)] = (
            _top_level_groupby,
            (ddf._name, p),
            cols,
            split_out,
            agg_cols,
            ("std" in agg_list or "var" in agg_list),
            on_host,
        )
        k = 0
        for c, col in enumerate(cols):
            for s in range(split_out[col]):
                dsk[(split_name, p, c, s)] = (getitem, (level_1_name, p), k)
                k += 1

    for c, col in enumerate(cols):
        for s in range(split_out[col]):
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
            [(level_2_name, c, s) for s in range(split_out[col])],
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


def _get_categories(ddf, cols, out_path, freq_limit, split_out, on_host):
    agg_cols = []
    agg_list = []
    return _groupby_to_disk(
        ddf, _write_uniques, cols, agg_cols, agg_list, out_path, freq_limit, split_out, on_host
    )


def _groupby_stats(ddf, cols, agg_cols, agg_list, out_path, freq_limit, split_out, on_host):
    if isinstance(agg_cols, str):
        agg_cols = [agg_cols]

    if agg_cols + agg_list == []:
        return _groupby_to_disk(
            ddf, _write_uniques, cols, agg_cols, agg_list, out_path, freq_limit, split_out, on_host
        )
    return _groupby_to_disk(
        ddf, _write_gb_stats, cols, agg_cols, agg_list, out_path, freq_limit, split_out, on_host
    )


def _get_cache():
    try:
        worker = get_worker()
    except ValueError:
        # There is no dask.distributed worker.
        # Assume client/worker are same process
        global DEFAULT_CACHE
        if DEFAULT_CACHE is None:
            DEFAULT_CACHE = CategoryCache()
        return DEFAULT_CACHE
    if not hasattr(worker, "cats_cache"):
        worker.cats_cache = CategoryCache()
    return worker.cats_cache


def _encode(name, path, gdf, cat_cache, na_sentinel=-1, freq_threshold=0):
    value = None
    if path:
        if cat_cache is not None:
            cat_cache = cat_cache.get(name, "disk")
            cache = _get_cache() if len(gdf) else None
            if cache:
                value = cache.fetch_data(name, path, cache=cat_cache, kind="cats")
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
        cat_cache = cat_cache.get(name, "disk")
        cache = _get_cache()
        if cache:
            return cache.fetch_data(name, path, cache=cat_cache, kind="stats")
    return cudf.io.read_parquet(path, index=False)
