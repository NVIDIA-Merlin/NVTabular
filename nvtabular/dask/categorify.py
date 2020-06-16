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

from io import BytesIO
from operator import getitem

import cudf
from cudf._lib.nvtx import annotate
from dask.base import tokenize
from dask.dataframe.core import _concat
from dask.distributed import get_worker
from dask.highlevelgraph import HighLevelGraph
from fsspec.core import get_fs_token_paths

try:
    import cupy as cp
except ImportError:
    import numpy as cp


class CategoryCache:
    def __init__(self):
        self.cat_cache = {}

    @annotate("get_categories", color="green", domain="nvt_python")
    def get_categories(self, col, path, cache="disk"):
        table = self.cat_cache.get(col, None)
        if table and not isinstance(table, cudf.DataFrame):
            df = cudf.io.read_parquet(table, index=False, columns=[col])
            df.index.name = "labels"
            df.reset_index(drop=False, inplace=True)
            return df

        if table is None:
            if cache in ("device", "disk"):
                table = cudf.io.read_parquet(path, index=False, columns=[col])
            elif cache == "host":
                with open(path, "rb") as f:
                    self.cat_cache[col] = BytesIO(f.read())
                table = cudf.io.read_parquet(self.cat_cache[col], index=False, columns=[col])
            table.index.name = "labels"
            table.reset_index(drop=False, inplace=True)
            if cache == "device":
                self.cat_cache[col] = table.copy(deep=False)
        return table


@annotate("cat_level_1", color="green", domain="nvt_python")
def _cat_level_1(gdf, columns, split_out, on_host):
    # First level of "catigorify"
    output = {}
    k = 0
    for i, col in enumerate(columns):
        gb = gdf[[col]].groupby(col, dropna=False).agg({col: ["count"]})
        gb.columns = gb.columns.get_level_values(1)
        gb.reset_index(drop=False, inplace=True)
        for j, split in enumerate(gb.partition_by_hash([col], split_out[col], keep_index=False)):
            if on_host:
                output[k] = split.to_pandas()
            else:
                output[k] = split
            k += 1
        gdf.drop(columns=[col], inplace=True)
        del gb
    del gdf
    return output


@annotate("cat_level_2", color="green", domain="nvt_python")
def _cat_level_2(dfs, col, freq_limit, on_host):
    ignore_index = True
    if on_host:
        # Pandas groupby does not have `dropna` arg
        gb = cudf.from_pandas(_concat(dfs, ignore_index)).groupby(col, dropna=False).sum()
    else:
        gb = _concat(dfs, ignore_index).groupby(col, dropna=False).sum()
    gb.reset_index(drop=False, inplace=True)
    if freq_limit:
        gb = gb[gb["count"] >= freq_limit]
    if on_host:
        gb_pd = gb[[col]].to_pandas()
        del gb
        return gb_pd
    return gb[[col]]


@annotate("cat_level_3", color="green", domain="nvt_python")
def _cat_level_3(dfs, base_path, col, on_host):
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


def _get_categories(ddf, cols, out_path, freq_limit, split_out, on_host):
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
        dsk[(level_1_name, p)] = (_cat_level_1, (ddf._name, p), cols, split_out, on_host)
        k = 0
        for c, col in enumerate(cols):
            for s in range(split_out[col]):
                dsk[(split_name, p, c, s)] = (getitem, (level_1_name, p), k)
                k += 1

    for c, col in enumerate(cols):
        for s in range(split_out[col]):
            dsk[(level_2_name, c, s)] = (
                _cat_level_2,
                [(split_name, p, c, s) for p in range(ddf.npartitions)],
                col,
                freq_limit,
                on_host,
            )

        dsk[(level_3_name, c)] = (
            _cat_level_3,
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


def _get_cache():
    try:
        worker = get_worker()
    except ValueError:
        # This is a metadata operation, so there is no "worker"
        # TODO: Handle metadata operations in a smarter way
        return None
    if not hasattr(worker, "cats_cache"):
        worker.cats_cache = CategoryCache()
    return worker.cats_cache


def _encode(name, path, gdf, cat_cache, na_sentinel=-1, freq_threshold=0):
    value = None
    if path:
        if cat_cache is not None:
            cat_cache = cat_cache.get(name, "disk")
            cache = _get_cache()
            if cache:
                value = cache.get_categories(name, path, cache=cat_cache)
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
