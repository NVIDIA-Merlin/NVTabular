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
import pyarrow as pa
from cudf.core.column import as_column, build_column
from cudf.io.parquet import ParquetWriter
from cudf.utils.dtypes import is_list_dtype
from dask.base import tokenize
from dask.core import flatten
from dask.dataframe.core import _concat
from dask.highlevelgraph import HighLevelGraph
from fsspec.core import get_fs_token_paths
from nvtx import annotate
from pyarrow import parquet as pq

from nvtabular.worker import fetch_table_data, get_worker_cache

from .groupby_statistics import GroupbyStatistics
from .operator import CAT
from .transform_operator import DFOperator


class Categorify(DFOperator):
    """
    Most of the data set will contain categorical features,
    and these variables are typically stored as text values.
    Machine Learning algorithms don't support these text values.
    Categorify operation can be added to the workflow to
    transform categorical features into unique integer values.

    Example usage::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add Categorify for categorical columns to the workflow
        proc.add_cat_preprocess(nvt.ops.Categorify(freq_threshold=10))

    Example with multi-hot::

        # Create toy dataframe
        df = cudf.DataFrame({
            'userID': [10001, 10002, 10003],
            'productID': [30003, 30005, 40005],
            'categories': [['Cat A', 'Cat B'], ['Cat C'], ['Cat A', 'Cat C', 'Cat D']],
            'label': [0,0,1]
        })

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=['userID', 'productID', 'categories'],
            cont_names=[],
            label_name=['label']
        )

        # Add Categorify for categorical columns to the workflow
        proc.add_preprocess(nvt.ops.Categorify())
        # Apply workflow
        proc.apply(nvt.Dataset(df), record_stats=True, output_path='./test/')

    Parameters
    -----------
    freq_threshold : int or dictionary:{column: freq_limit_value}, default 0
        Categories with a count/frequency below this threshold will be
        ommited from the encoding and corresponding data will be mapped
        to the "null" category. Can be represented as both an integer or
        a dictionary with column names as keys and frequency limit as
        value. If dictionary is used, all columns targeted must be included
        in the dictionary.
    columns : list of str or list(str), default None
        Categorical columns (or multi-column "groups") to target for this op.
        If None, the operation will target all known categorical columns.
        If columns contains 1+ list(str) elements, the columns within each
        list/group will be encoded according to the `encode_type` setting.
    encode_type : {"joint", "combo"}, default "joint"
        If "joint", the columns within any multi-column group will be
        jointly encoded. If "combo", the combination of values will be
        encoded as a new column. Note that replacement is not allowed for
        "combo", because the same column name can be included in
        multiple groups.
    replace : bool, default True
        Replaces the transformed column with the original input.
        Note that this does not apply to multi-column groups with
        `encoded_type="combo"`.
    tree_width : dict or int, optional
        Passed to `GroupbyStatistics` dependency.
    out_path : str, optional
        Passed to `GroupbyStatistics` dependency.
    on_host : bool, default True
        Passed to `GroupbyStatistics` dependency.
    na_sentinel : default 0
        Label to use for null-category mapping
    cat_cache : {"device", "host", "disk"} or dict
        Location to cache the list of unique categories for
        each categorical column. If passing a dict, each key and value
        should correspond to the column name and location, respectively.
        Default is "host" for all columns.
    dtype :
        If specified, categorical labels will be cast to this dtype
        after encoding is performed.
    name_sep : str, default "_"
        String separator to use between concatenated column names
        for multi-column groups.
    search_sorted : bool, default False.
        Set it True to apply searchsorted algorithm in encoding.
    """

    default_in = CAT
    default_out = CAT

    def __init__(
        self,
        freq_threshold=0,
        columns=None,
        replace=True,
        out_path=None,
        tree_width=None,
        na_sentinel=None,
        cat_cache="host",
        dtype=None,
        on_host=True,
        encode_type="joint",
        name_sep="_",
        search_sorted=False,
    ):

        # We need to handle three types of encoding here:
        #
        #   (1) Conventional encoding. There are no multi-column groups. So,
        #       each categorical column is separately transformed into a new
        #       "encoded" column (1-to-1).  The unique values are calculated
        #       separately for each column.
        #
        #   (2) Multi-column "Joint" encoding (there are multi-column groups
        #       in `columns` and `encode_type="joint"`).  Still a
        #       1-to-1 transofrmation of categorical columns.  However,
        #       we concatenate column groups to determine uniques (rather
        #       than getting uniques of each categorical column separately).
        #
        #   (3) Multi-column "Group" encoding (there are multi-column groups
        #       in `columns` and `encode_type="combo"`). No longer
        #       a 1-to-1 transformation of categorical columns. Each column
        #       group will be transformed to a single "encoded" column.  This
        #       means the unique "values" correspond to unique combinations.
        #       Since the same column may be included in multiple groups,
        #       replacement is not allowed for this transform.

        # Set column_groups if the user has passed in a list of columns.
        # The purpose is to capture multi-column groups. If the user doesn't
        # specify `columns`, there are no multi-column groups to worry about.
        self.column_groups = None
        self.name_sep = name_sep

        # For case (2), we need to keep track of the multi-column group name
        # that will be used for the joint encoding of each column in that group.
        # For case (3), we also use this "storage name" to signify the name of
        # the file with the required "combination" groupby statistics.
        self.storage_name = {}
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(columns, list):
            # User passed in a list of column groups. We need to figure out
            # if this list contains any multi-column groups, and if there
            # are any (obvious) problems with these groups
            self.column_groups = columns
            columns = list(set(flatten(columns, container=list)))
            columns_all = list(flatten(columns, container=list))
            if sorted(columns_all) != sorted(columns) and encode_type == "joint":
                # If we are doing "joint" encoding, there must be unique mapping
                # between input column names and column groups.  Otherwise, more
                # than one unique-value table could be used to encode the same
                # column.
                raise ValueError("Same column name included in multiple groups.")
            for group in self.column_groups:
                if isinstance(group, list) and len(group) > 1:
                    # For multi-column groups, we concatenate column names
                    # to get the "group" name.
                    name = _make_name(*group, sep=self.name_sep)
                    for col in group:
                        self.storage_name[col] = name

        # Only support two kinds of multi-column encoding
        if encode_type not in ("joint", "combo"):
            raise ValueError(f"encode_type={encode_type} not supported.")

        # Other self-explanatory intialization
        super().__init__(columns=columns, replace=replace)
        self.freq_threshold = freq_threshold or 0
        self.out_path = out_path or "./"
        self.tree_width = tree_width
        self.na_sentinel = na_sentinel or 0
        self.dtype = dtype
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.stat_name = "categories"
        self.encode_type = encode_type
        self.search_sorted = search_sorted

        if self.search_sorted and self.freq_threshold:
            raise ValueError(
                "cannot use search_sorted=True with anything else than the default freq_threshold"
            )

    @property
    def req_stats(self):
        return [
            GroupbyStatistics(
                columns=self.column_groups or self.columns,
                concat_groups=self.encode_type == "joint",
                cont_names=[],
                stats=[],
                freq_threshold=self.freq_threshold,
                tree_width=self.tree_width,
                out_path=self.out_path,
                on_host=self.on_host,
                stat_name=self.stat_name,
                name_sep=self.name_sep,
            )
        ]

    @annotate("Categorify_op", color="darkgreen", domain="nvt_python")
    def apply_op(
        self,
        gdf: cudf.DataFrame,
        columns_ctx: dict,
        input_cols,
        target_cols=["base"],
        stats_context={},
    ):
        new_gdf = gdf.copy(deep=False)
        target_columns = self.get_columns(columns_ctx, input_cols, target_cols)
        if isinstance(self.freq_threshold, dict):
            assert all(x in self.freq_threshold for x in target_columns)
        if not target_columns:
            return new_gdf

        if self.column_groups and not self.encode_type == "joint":
            # Case (3) - We want to track multi- and single-column groups separately
            #            when we are NOT performing a joint encoding. This is because
            #            there is not a 1-to-1 mapping for columns in multi-col groups.
            #            We use `multi_col_group` to preserve the list format of
            #            multi-column groups only, and use `cat_names` to store the
            #            string representation of both single- and multi-column groups.
            #
            cat_names, multi_col_group = _get_multicolumn_names(
                self.column_groups, gdf.columns, self.name_sep
            )
        else:
            # Case (1) & (2) - Simple 1-to-1 mapping
            multi_col_group = {}
            cat_names = [name for name in target_columns if name in gdf.columns]

        # Encode each column-group separately
        for name in cat_names:
            new_col = f"{name}_{self._id}"

            # Use the column-group `list` directly (not the string name)
            use_name = multi_col_group.get(name, name)
            # Storage name may be different than group for case (2)
            # Only use the "aliased" `storage_name` if we are dealing with
            # a multi-column group, or if we are doing joint encoding
            if use_name != name or self.encode_type == "joint":
                storage_name = self.storage_name.get(name, name)
            else:
                storage_name = name
            path = stats_context[self.stat_name][storage_name]
            if not self.column_groups and _is_list_col([name], gdf):
                if "mh" not in columns_ctx["categorical"]:
                    columns_ctx["categorical"]["mh"] = []
                if name not in columns_ctx["categorical"]["mh"]:
                    columns_ctx["categorical"]["mh"].append(name)
            new_gdf[new_col] = _encode(
                use_name,
                storage_name,
                path,
                gdf,
                self.cat_cache,
                na_sentinel=self.na_sentinel,
                freq_threshold=self.freq_threshold[name]
                if isinstance(self.freq_threshold, dict)
                else self.freq_threshold,
                search_sorted=self.search_sorted,
            )
            if self.dtype:
                new_gdf[new_col] = new_gdf[new_col].astype(self.dtype, copy=False)

        # Deal with replacement
        if self.replace:
            for name in cat_names:
                new_col = f"{name}_{self._id}"
                new_gdf[name] = new_gdf[new_col]
                new_gdf.drop(columns=[new_col], inplace=True)

        self.update_columns_ctx(columns_ctx, input_cols, new_gdf.columns, target_columns)
        return new_gdf


def _get_embedding_order(cat_names):
    """Returns a consistent sorder order for categorical variables

    Parameters
    -----------
    cat_names : list of str
        names of the categorical columns
    """
    return sorted(cat_names)


def get_embedding_sizes(workflow):
    mh_cols = None
    cols = _get_embedding_order(workflow.columns_ctx["categorical"]["base"])
    if "mh" in workflow.columns_ctx["categorical"]:
        mh_cols = _get_embedding_order(workflow.columns_ctx["categorical"]["mh"])
        for col in mh_cols:
            cols.remove(col)
    res = _get_embeddings_dask(workflow.stats["categories"], cols)
    if mh_cols:
        res = res, _get_embeddings_dask(workflow.stats["categories"], mh_cols)
    return res


def _get_embeddings_dask(paths, cat_names):
    embeddings = {}
    for col in cat_names:
        path = paths[col]
        num_rows, _, _ = cudf.io.read_parquet_metadata(path)
        embeddings[col] = _emb_sz_rule(num_rows)
    return embeddings


def _emb_sz_rule(n_cat: int) -> int:
    return n_cat, int(min(16, round(1.6 * n_cat ** 0.56)))


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
        if _is_list_col(cat_col_group, df_gb):
            # handle list columns by encoding the list values
            df_gb = cudf.DataFrame({cat_col_group[0]: df_gb[cat_col_group[0]].list.leaves})

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
                output[k] = split.to_arrow(preserve_index=False)
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

    if on_host:
        df = pa.concat_tables(dfs, promote=True)
        df = cudf.DataFrame.from_arrow(df)
    else:
        df = _concat(dfs, ignore_index=True)
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
        gb_pd = gb[required].to_arrow(preserve_index=False)
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
    if isinstance(col_group, str):
        col_group = [col_group]

    rel_path = "cat_stats.%s.parquet" % (_make_name(*col_group, sep=name_sep))
    path = os.path.join(base_path, rel_path)
    pwriter = None
    if not on_host:
        pwriter = ParquetWriter(path, compression=None)

    # Loop over dfs and append to file
    # TODO: For high-cardinality columns, should support
    #       Dask-based to_parquet call here (but would need to
    #       support directory reading within dependent ops)
    n_writes = 0
    for df in dfs:
        if len(df):
            if on_host:
                # Use pyarrow - df is already a pyarrow table
                if pwriter is None:
                    pwriter = pq.ParquetWriter(path, df.schema, compression=None)
                pwriter.write_table(df)
            else:
                # Use CuDF
                df.reset_index(drop=True, inplace=True)
                pwriter.write_table(df)
            n_writes += 1

    # No data to write
    if n_writes == 0:
        raise RuntimeError("GroupbyStatistics result is empty.")

    # Close writer and return path
    pwriter.close()

    return path


@annotate("write_uniques", color="green", domain="nvt_python")
def _write_uniques(dfs, base_path, col_group, on_host, concat_groups, name_sep):
    if concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=name_sep)]
    if isinstance(col_group, str):
        col_group = [col_group]
    if on_host:
        df = pa.concat_tables(dfs, promote=True)
        df = cudf.DataFrame.from_arrow(df)
    else:
        df = _concat(dfs, ignore_index=True)
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
                    [cudf.Series([None], dtype=df[col].dtype), df[col]],
                    ignore_index=True,
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


def _encode(
    name,
    storage_name,
    path,
    gdf,
    cat_cache,
    na_sentinel=-1,
    freq_threshold=0,
    search_sorted=False,
):
    value = None
    selection_l = name if isinstance(name, list) else [name]
    selection_r = name if isinstance(name, list) else [storage_name]
    list_col = _is_list_col(selection_l, gdf)
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

    if not search_sorted:
        if list_col:
            codes = cudf.DataFrame({selection_l[0]: gdf[selection_l[0]].list.leaves})
            codes["order"] = cp.arange(len(codes))
        else:
            codes = cudf.DataFrame({"order": cp.arange(len(gdf))})
            for c in selection_l:
                codes[c] = gdf[c].copy()
        labels = codes.merge(
            value, left_on=selection_l, right_on=selection_r, how="left"
        ).sort_values("order")["labels"]
        labels.fillna(na_sentinel, inplace=True)
        labels = labels.values
    else:
        # Use `searchsorted` if we are using a "full" encoding
        if list_col:
            labels = value[selection_r].searchsorted(
                gdf[selection_l[0]].list.leaves, side="left", na_position="first"
            )
        else:
            labels = value[selection_r].searchsorted(
                gdf[selection_l], side="left", na_position="first"
            )
        labels[labels >= len(value[selection_r])] = na_sentinel

    if list_col:
        labels = _encode_list_column(gdf[selection_l[0]], labels)

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


def _is_list_col(column_group, df):
    has_lists = any(is_list_dtype(df[col]) for col in column_group)
    if has_lists and len(column_group) != 1:
        raise ValueError("Can't categorical encode multiple list columns")
    return has_lists


def _encode_list_column(original, encoded):
    encoded = as_column(encoded)
    return build_column(
        None,
        dtype=cudf.core.dtypes.ListDtype(encoded.dtype),
        size=original.size,
        children=(original._column.offsets, encoded),
    )
