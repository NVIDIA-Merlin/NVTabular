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

import os
import warnings
from operator import getitem

import cudf
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.base import tokenize
from dask.core import flatten
from dask.dataframe.core import _concat
from dask.dataframe.shuffle import shuffle_group
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from fsspec.core import get_fs_token_paths
from nvtx import annotate
from pyarrow import parquet as pq

from nvtabular.dispatch import (
    DataFrameType,
    _arange,
    _encode_list_column,
    _flatten_list_column,
    _hash_series,
    _is_list_dtype,
    _parquet_writer_dispatch,
    _read_parquet_dispatch,
    _series_has_nulls,
)
from nvtabular.worker import fetch_table_data, get_worker_cache

from .operator import ColumnNames, Operator
from .stat_operator import StatOperator


class Categorify(StatOperator):
    """
    Most of the data set will contain categorical features,
    and these variables are typically stored as text values.
    Machine Learning algorithms don't support these text values.
    Categorify operation can be added to the workflow to
    transform categorical features into unique integer values.

    Example usage::

        # Define pipeline
        cat_features = CATEGORICAL_COLUMNS >> nvt.ops.Categorify(freq_threshold=10)

        # Initialize the workflow and execute it
        proc = nvt.Workflow(cat_features)
        proc.fit(dataset)
        proc.transform(dataset).to_parquet('./test/')

    Example for frequency hashing::

        import cudf
        import nvtabular as nvt

        # Create toy dataset
        df = cudf.DataFrame({
            'author': ['User_A', 'User_B', 'User_C', 'User_C', 'User_A', 'User_B', 'User_A'],
            'productID': [100, 101, 102, 101, 102, 103, 103],
            'label': [0, 0, 1, 1, 1, 0, 0]
        })
        dataset = nvt.Dataset(df)

        # Define pipeline
        CATEGORICAL_COLUMNS = ['author', 'productID']
        cat_features = CATEGORICAL_COLUMNS >> nvt.ops.Categorify(
            freq_threshold={"author": 3, "productID": 2},
            num_buckets={"author": 10, "productID": 20})


        # Initialize the workflow and execute it
        proc = nvt.Workflow(cat_features)
        proc.fit(dataset)
        ddf = proc.transform(dataset).to_ddf()

        # Print results
        print(ddf.compute())

    Example with multi-hot::

        import cudf
        import nvtabular as nvt

        # Create toy dataset
        df = cudf.DataFrame({
            'userID': [10001, 10002, 10003],
            'productID': [30003, 30005, 40005],
            'categories': [['Cat A', 'Cat B'], ['Cat C'], ['Cat A', 'Cat C', 'Cat D']],
            'label': [0,0,1]
        })
        dataset = nvt.Dataset(df)

        # Define pipeline
        CATEGORICAL_COLUMNS = ['userID', 'productID', 'categories']
        cat_features = CATEGORICAL_COLUMNS >> nvt.ops.Categorify()

        # Initialize the workflow and execute it
        proc = nvt.Workflow(cat_features)
        proc.fit(dataset)
        ddf = proc.transform(dataset).to_ddf()

        # Print results
        print(ddf.compute())

    Parameters
    -----------
    freq_threshold : int or dictionary:{column: freq_limit_value}, default 0
        Categories with a count/frequency below this threshold will be
        ommited from the encoding and corresponding data will be mapped
        to the "null" category. Can be represented as both an integer or
        a dictionary with column names as keys and frequency limit as
        value. If dictionary is used, all columns targeted must be included
        in the dictionary.
    encode_type : {"joint", "combo"}, default "joint"
        If "joint", the columns within any multi-column group will be
        jointly encoded. If "combo", the combination of values will be
        encoded as a new column. Note that replacement is not allowed for
        "combo", because the same column name can be included in
        multiple groups.
    tree_width : dict or int, optional
        Tree width of the hash-based groupby reduction for each categorical
        column. High-cardinality columns may require a large `tree_width`,
        while low-cardinality columns can likely use `tree_width=1`.
        If passing a dict, each key and value should correspond to the column
        name and width, respectively. The default value is 8 for all columns.
    out_path : str, optional
        Root directory where groupby statistics will be written out in
        parquet format.
    on_host : bool, default True
        Whether to convert cudf data to pandas between tasks in the hash-based
        groupby reduction. The extra host <-> device data movement can reduce
        performance.  However, using `on_host=True` typically improves stability
        (by avoiding device-level memory pressure).
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
    num_buckets : int, or dictionary:{column: num_hash_buckets}
        Column-wise modulo to apply after hash function. Note that this
        means that the corresponding value will be the categorical cardinality
        of the transformed categorical feature. If given as an int, that value
        will be used as the number of "hash buckets" for every feature. If a dictionary is passed,
        it will be used to specify explicit mappings from a column name to a number of buckets.
        In this case, only the columns specified in the keys of `num_buckets`
        will be transformed.
    max_size : int or dictionary:{column: max_size_value}, default 0
        This parameter allows you to set the maximum size for an embedding table for each column.
        For example, if max_size is set to 1000 only the first 999 most frequent values for each
        column will be be encoded, and the rest will be mapped to a single value (0). To map the
        rest to a number of buckets,  you can set the num_buckets parameter > 1. In that case, topK
        value will be `max_size - num_buckets -1`.  Setting the max_size param means that
        freq_threshold should not be given.  If the num_buckets parameter is set,  it must be
        smaller than the max_size value.
    """

    def __init__(
        self,
        freq_threshold=0,
        out_path=None,
        tree_width=None,
        na_sentinel=None,
        cat_cache="host",
        dtype=None,
        on_host=True,
        encode_type="joint",
        name_sep="_",
        search_sorted=False,
        num_buckets=None,
        max_size=0,
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

        # Only support two kinds of multi-column encoding
        if encode_type not in ("joint", "combo"):
            raise ValueError(f"encode_type={encode_type} not supported.")

        # Other self-explanatory intialization
        super().__init__()
        self.freq_threshold = freq_threshold or 0
        self.out_path = out_path or "./"
        self.tree_width = tree_width
        self.na_sentinel = na_sentinel or 0
        self.dtype = dtype
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.encode_type = encode_type
        self.search_sorted = search_sorted
        self.categories = {}
        self.mh_columns = []

        if self.search_sorted and self.freq_threshold:
            raise ValueError(
                "cannot use search_sorted=True with anything else than the default freq_threshold"
            )
        if num_buckets == 0:
            raise ValueError(
                "For hashing num_buckets should be an int > 1, otherwise set num_buckets=None."
            )
        elif isinstance(num_buckets, dict):
            self.num_buckets = num_buckets
        elif isinstance(num_buckets, int) or num_buckets is None:
            self.num_buckets = num_buckets
        else:
            raise ValueError(
                "`num_buckets` must be dict or int, got type {}".format(type(num_buckets))
            )
        if isinstance(max_size, dict):
            self.max_size = max_size
        elif isinstance(max_size, int) or max_size is None:
            self.max_size = max_size
        else:
            raise ValueError("max_size must be dict or int, got type {}".format(type(max_size)))
        if freq_threshold and max_size:
            raise ValueError("cannot use freq_threshold param together with max_size param")

        if self.num_buckets is not None:
            # See: nvtabular.dispatch._hash_series
            warnings.warn(
                "Performing a hash-based transformation. Do not "
                "expect Categorify to be consistent on GPU and CPU "
                "with this num_buckets setting!"
            )

    @annotate("Categorify_fit", color="darkgreen", domain="nvt_python")
    def fit(self, columns: ColumnNames, ddf: dd.DataFrame):
        # User passed in a list of column groups. We need to figure out
        # if this list contains any multi-column groups, and if there
        # are any (obvious) problems with these groups
        columns_uniq = list(set(flatten(columns, container=tuple)))
        columns_all = list(flatten(columns, container=tuple))
        if sorted(columns_all) != sorted(columns_uniq) and self.encode_type == "joint":
            # If we are doing "joint" encoding, there must be unique mapping
            # between input column names and column groups.  Otherwise, more
            # than one unique-value table could be used to encode the same
            # column.
            raise ValueError("Same column name included in multiple groups.")

        for group in columns:
            if isinstance(group, tuple) and len(group) > 1:
                # For multi-column groups, we concatenate column names
                # to get the "group" name.
                name = _make_name(*group, sep=self.name_sep)
                for col in group:
                    self.storage_name[col] = name

        # Check metadata type to reset on_host and cat_cache if the
        # underlying ddf is already a pandas-backed collection
        if isinstance(ddf._meta, pd.DataFrame):
            self.on_host = False
            # Cannot use "device" caching if the data is pandas-backed
            self.cat_cache = "host" if self.cat_cache == "device" else self.cat_cache
            if self.search_sorted:
                # Pandas' search_sorted only works with Series.
                # For now, it is safest to disallow this option.
                self.search_sorted = False
                warnings.warn("Cannot use `search_sorted=True` for pandas-backed data.")

        # convert tuples to lists
        columns = [list(c) if isinstance(c, tuple) else c for c in columns]
        dsk, key = _category_stats(
            ddf,
            columns,
            [],
            [],
            self.out_path,
            self.freq_threshold,
            self.tree_width,
            self.on_host,
            concat_groups=self.encode_type == "joint",
            name_sep=self.name_sep,
            max_size=self.max_size,
            num_buckets=self.num_buckets,
        )
        # TODO: we can't check the dtypes on the ddf here since they are incorrect
        # for cudf's list type. So, we're checking the first partition. fix.
        return Delayed(key, dsk), ddf.partitions[0].map_partitions(lambda df: _is_list_dtype(df))

    def fit_finalize(self, dask_stats):
        _col_is_list = dask_stats[1]
        self.mh_columns = [
            col for col, _is_list in zip(_col_is_list.index, _col_is_list) if _is_list
        ]
        categories = dask_stats[0]
        for col in categories:
            self.categories[col] = categories[col]

    def clear(self):
        self.categories = {}
        self.mh_columns = []

    def set_storage_path(self, new_path, copy=False):
        self.categories = _copy_storage(self.categories, self.out_path, new_path, copy=copy)
        self.out_path = new_path

    @annotate("Categorify_transform", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        new_df = df.copy(deep=False)
        if isinstance(self.freq_threshold, dict):
            assert all(x in self.freq_threshold for x in columns)

        if self.encode_type == "combo":
            # Case (3) - We want to track multi- and single-column groups separately
            #            when we are NOT performing a joint encoding. This is because
            #            there is not a 1-to-1 mapping for columns in multi-col groups.
            #            We use `multi_col_group` to preserve the list format of
            #            multi-column groups only, and use `cat_names` to store the
            #            string representation of both single- and multi-column groups.
            #
            cat_names, multi_col_group = _get_multicolumn_names(columns, df.columns, self.name_sep)
        else:
            # Case (1) & (2) - Simple 1-to-1 mapping
            multi_col_group = {}
            cat_names = list(flatten(columns, container=tuple))

        # Encode each column-group separately
        for name in cat_names:
            try:
                # Use the column-group `list` directly (not the string name)
                use_name = multi_col_group.get(name, name)

                # Storage name may be different than group for case (2)
                # Only use the "aliased" `storage_name` if we are dealing with
                # a multi-column group, or if we are doing joint encoding

                if use_name != name or self.encode_type == "joint":
                    storage_name = self.storage_name.get(name, name)
                else:
                    storage_name = name

                if isinstance(use_name, tuple):
                    use_name = list(use_name)

                path = self.categories[storage_name]
                new_df[name] = _encode(
                    use_name,
                    storage_name,
                    path,
                    df,
                    self.cat_cache,
                    na_sentinel=self.na_sentinel,
                    freq_threshold=self.freq_threshold[name]
                    if isinstance(self.freq_threshold, dict)
                    else self.freq_threshold,
                    search_sorted=self.search_sorted,
                    buckets=self.num_buckets,
                    encode_type=self.encode_type,
                    cat_names=cat_names,
                    max_size=self.max_size,
                )
                if self.dtype:
                    new_df[name] = new_df[name].astype(self.dtype, copy=False)
            except Exception as e:
                raise RuntimeError(f"Failed to categorical encode column {name}") from e

        return new_df

    def output_column_names(self, columns: ColumnNames) -> ColumnNames:
        if self.encode_type == "combo":
            cat_names, _ = _get_multicolumn_names(columns, columns, self.name_sep)
            return cat_names
        return list(flatten(columns, container=tuple))

    def get_embedding_sizes(self, columns):
        return _get_embeddings_dask(
            self.categories, columns, self.num_buckets, self.freq_threshold, self.max_size
        )

    def get_multihot_columns(self):
        return self.mh_columns

    transform.__doc__ = Operator.transform.__doc__
    fit.__doc__ = StatOperator.fit.__doc__
    fit_finalize.__doc__ = StatOperator.fit_finalize.__doc__


def _get_embedding_order(cat_names):
    """Returns a consistent sorder order for categorical variables

    Parameters
    -----------
    cat_names : list of str
        names of the categorical columns
    """
    return sorted(cat_names)


def get_embedding_sizes(workflow):
    """ Returns a dictionary of best embedding sizes from the workflow """
    # TODO: do we need to distinguish multihot columns here?  (if so why? )
    queue = [workflow.column_group]
    output = {}
    multihot_columns = set()
    while queue:
        current = queue.pop()
        if current.op and hasattr(current.op, "get_embedding_sizes"):
            output.update(current.op.get_embedding_sizes(current.columns))

            if hasattr(current.op, "get_multihot_columns"):
                multihot_columns.update(current.op.get_multihot_columns())

        elif not current.op:
            # only follow parents if its not an operator node (which could
            # transform meaning of the get_embedding_sizes
            queue.extend(current.parents)

    # TODO: returning differnt return types like this (based off the presence
    # of multihot features) is pretty janky. fix.
    if not multihot_columns:
        return output

    single_hots = {k: v for k, v in output.items() if k not in multihot_columns}
    multi_hots = {k: v for k, v in output.items() if k in multihot_columns}
    return single_hots, multi_hots


def _get_embeddings_dask(paths, cat_names, buckets=0, freq_limit=0, max_size=0):
    embeddings = {}
    if isinstance(freq_limit, int):
        freq_limit = {name: freq_limit for name in cat_names}
    if isinstance(buckets, int):
        buckets = {name: buckets for name in cat_names}
    if isinstance(max_size, int):
        max_size = {name: max_size for name in cat_names}
    for col in _get_embedding_order(cat_names):
        path = paths.get(col)
        num_rows = pq.ParquetFile(path).metadata.num_rows if path else 0

        if not buckets:
            bucket_size = 0
        else:
            bucket_size = buckets.get(col, 0)
        if bucket_size and not freq_limit[col] and not max_size[col]:
            # pure hashing (no categorical lookup)
            num_rows = bucket_size
        else:
            num_rows += bucket_size

        embeddings[col] = _emb_sz_rule(num_rows)
    return embeddings


def _emb_sz_rule(n_cat: int, minimum_size=16, maximum_size=512) -> int:
    return n_cat, min(max(minimum_size, round(1.6 * n_cat ** 0.56)), maximum_size)


def _make_name(*args, sep="_"):
    return sep.join(args)


@annotate("top_level_groupby", color="green", domain="nvt_python")
def _top_level_groupby(
    df, cat_col_groups, tree_width, cont_cols, agg_list, on_host, concat_groups, name_sep
):
    sum_sq = "std" in agg_list or "var" in agg_list
    calculate_min = "min" in agg_list
    calculate_max = "max" in agg_list

    # Top-level operation for category-based groupby aggregations
    output = {}
    k = 0
    for i, cat_col_group in enumerate(cat_col_groups):
        if isinstance(cat_col_group, tuple):
            cat_col_group = list(cat_col_group)

        if isinstance(cat_col_group, str):
            cat_col_group = [cat_col_group]
        cat_col_group_str = _make_name(*cat_col_group, sep=name_sep)

        if concat_groups and len(cat_col_group) > 1:
            # Concatenate columns and replace cat_col_group
            # with the single name
            df_gb = type(df)()
            ignore_index = True
            df_gb[cat_col_group_str] = _concat([df[col] for col in cat_col_group], ignore_index)
            cat_col_group = [cat_col_group_str]
        else:
            # Compile aggregation dictionary and add "squared-sum"
            # column(s) (necessary when `cont_cols` is non-empty)
            df_gb = df[cat_col_group + cont_cols].copy(deep=False)

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
        # (flattening provides better cudf/pd support)
        if _is_list_col(cat_col_group, df_gb):
            # handle list columns by encoding the list values
            df_gb = _flatten_list_column(df_gb[cat_col_group[0]])

        # NOTE: groupby(..., dropna=False) requires pandas>=1.1.0
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
        nsplits = tree_width[cat_col_group_str]
        for j, split in shuffle_group(
            gb, cat_col_group, 0, nsplits, nsplits, True, nsplits
        ).items():
            if on_host:
                output[k] = split.to_arrow(preserve_index=False)
            else:
                output[k] = split
            k += 1
        del gb
    return output


@annotate("mid_level_groupby", color="green", domain="nvt_python")
def _mid_level_groupby(
    dfs, col_group, cont_cols, agg_list, freq_limit, on_host, concat_groups, name_sep, max_emb_size
):
    if isinstance(col_group, str):
        col_group = [col_group]
    elif isinstance(col_group, tuple):
        col_group = list(col_group)

    if concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=name_sep)]

    if on_host:
        # Construct gpu DataFrame from pyarrow data.
        # `on_host=True` implies gpu-backed data.
        df = pa.concat_tables(dfs, promote=True)
        df = cudf.DataFrame.from_arrow(df)
    else:
        df = _concat(dfs, ignore_index=True)
    groups = df.groupby(col_group, dropna=False)
    gb = groups.agg({col: _get_aggregation_type(col) for col in df.columns if col not in col_group})
    gb.reset_index(drop=False, inplace=True)

    name_count = _make_name(*(col_group + ["count"]), sep=name_sep)
    if freq_limit and not max_emb_size:
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
def _write_gb_stats(
    dfs, base_path, col_group, on_host, concat_groups, name_sep, max_emb_size=None, nbuckets=None
):
    if concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=name_sep)]
    if isinstance(col_group, str):
        col_group = [col_group]

    rel_path = "cat_stats.%s.parquet" % (_make_name(*col_group, sep=name_sep))
    path = os.path.join(base_path, rel_path)
    pwriter = None
    if not on_host and len(dfs):
        pwriter = _parquet_writer_dispatch(dfs[0])(path, compression=None)

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
    if pwriter is not None:
        pwriter.close()

    return path


@annotate("write_uniques", color="green", domain="nvt_python")
def _write_uniques(
    dfs, base_path, col_group, on_host, concat_groups, name_sep, max_emb_size=None, nbuckets=None
):
    if concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=name_sep)]
    if isinstance(col_group, str):
        col_group = [col_group]
    if on_host:
        # Construct gpu DataFrame from pyarrow data.
        # `on_host=True` implies gpu-backed data.
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
            name_count = col + "_count"
            if max_emb_size:
                if isinstance(max_emb_size, int):
                    max_emb_size = {col: max_emb_size}
                if nbuckets:
                    if isinstance(nbuckets, int):
                        nlargest = max_emb_size[col] - nbuckets - 1
                    else:
                        nlargest = max_emb_size[col] - nbuckets[col] - 1
                else:
                    nlargest = max_emb_size[col] - 1

                if nlargest <= 0:
                    raise ValueError("`nlargest` cannot be 0 or negative")

                if nlargest < len(df):
                    df = df.nlargest(n=nlargest, columns=name_count)
            if not _series_has_nulls(df[col]):
                nulls_missing = True
                new_cols[col] = _concat(
                    [df._constructor_sliced([None], dtype=df[col].dtype), df[col]],
                    ignore_index=True,
                )
            else:
                new_cols[col] = df[col].copy(deep=False)
        if nulls_missing:
            df = type(df)(new_cols)
        df.to_parquet(path, index=False, compression=None)
    else:
        df_null = type(df)({c: [None] for c in col_group})
        for c in col_group:
            df_null[c] = df_null[c].astype(df[c].dtype)
        df_null.to_parquet(path, index=False, compression=None)
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
    max_size=None,
    nbuckets=None,
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
        if isinstance(col, tuple):
            col = list(col)

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
                max_size,
            )

        dsk[(level_3_name, c)] = (
            write_func,
            [(level_2_name, c, s) for s in range(tree_width[col_str])],
            out_path,
            col,
            on_host,
            concat_groups,
            name_sep,
            max_size,
            nbuckets,
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
    max_size=None,
    num_buckets=None,
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
            max_size=max_size,
            nbuckets=num_buckets,
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
        max_size=max_size,
        nbuckets=num_buckets,
    )


def _encode(
    name,
    storage_name,
    path,
    df,
    cat_cache,
    na_sentinel=-1,
    freq_threshold=0,
    search_sorted=False,
    buckets=None,
    encode_type="joint",
    cat_names=None,
    max_size=0,
):
    if isinstance(buckets, int):
        buckets = {name: buckets for name in cat_names}
    # this is to apply freq_hashing logic
    if max_size:
        freq_threshold = 1
    value = None
    selection_l = name if isinstance(name, list) else [name]
    selection_r = name if isinstance(name, list) else [storage_name]
    list_col = _is_list_col(selection_l, df)
    if path:
        read_pq_func = _read_parquet_dispatch(df)
        if cat_cache is not None:
            cat_cache = (
                cat_cache if isinstance(cat_cache, str) else cat_cache.get(storage_name, "disk")
            )
            if len(df):
                with get_worker_cache("cats") as cache:
                    value = fetch_table_data(
                        cache,
                        path,
                        columns=selection_r,
                        cache=cat_cache,
                        cats_only=True,
                        reader=read_pq_func,
                    )
        else:
            value = read_pq_func(path, columns=selection_r)
            value.index.name = "labels"
            value.reset_index(drop=False, inplace=True)

    if value is None:
        value = type(df)()
        for c in selection_r:
            typ = df[selection_l[0]].dtype if len(selection_l) == 1 else df[c].dtype
            value[c] = df._constructor_sliced([None], dtype=typ)
        value.index.name = "labels"
        value.reset_index(drop=False, inplace=True)

    if not search_sorted:
        if list_col:
            codes = _flatten_list_column(df[selection_l[0]])
            codes["order"] = _arange(len(codes), like_df=df)
        else:
            codes = type(df)({"order": _arange(len(df), like_df=df)}, index=df.index)
            for c in selection_l:
                codes[c] = df[c].copy()
        if buckets and storage_name in buckets:
            na_sentinel = _hash_bucket(df, buckets, selection_l, encode_type=encode_type)
        # apply frequency hashing
        if freq_threshold and buckets and storage_name in buckets:
            merged_df = codes.merge(
                value, left_on=selection_l, right_on=selection_r, how="left"
            ).sort_values("order")
            merged_df.reset_index(drop=True, inplace=True)
            max_id = merged_df["labels"].max()
            merged_df["labels"].fillna(
                df._constructor_sliced(na_sentinel + max_id + 1), inplace=True
            )
            labels = merged_df["labels"].values
        # only do hashing
        elif buckets and storage_name in buckets:
            labels = na_sentinel
        # no hashing
        else:
            na_sentinel = 0
            labels = codes.merge(
                value, left_on=selection_l, right_on=selection_r, how="left"
            ).sort_values("order")["labels"]
            labels.fillna(na_sentinel, inplace=True)
            labels = labels.values
    else:
        # Use `searchsorted` if we are using a "full" encoding
        if list_col:
            labels = value[selection_r].searchsorted(
                df[selection_l[0]].list.leaves, side="left", na_position="first"
            )
        else:
            labels = value[selection_r].searchsorted(
                df[selection_l], side="left", na_position="first"
            )
        labels[labels >= len(value[selection_r])] = na_sentinel

    if list_col:
        labels = _encode_list_column(df[selection_l[0]], labels)

    return labels


def _read_groupby_stat_df(path, name, cat_cache, read_pq_func):
    if cat_cache is not None:
        cat_cache = cat_cache if isinstance(cat_cache, str) else cat_cache.get(name, "disk")
        with get_worker_cache("stats") as cache:
            if cache:
                return fetch_table_data(cache, path, cache=cat_cache, reader=read_pq_func)
    return read_pq_func(path, index=False)


def _get_multicolumn_names(column_groups, df_columns, name_sep):
    cat_names = []
    multi_col_group = {}
    for col_group in column_groups:
        if isinstance(col_group, (list, tuple)):
            name = _make_name(*col_group, sep=name_sep)
            if name not in cat_names:
                cat_names.append(name)
                # TODO: Perhaps we should check that all columns from the group
                #       are in df here?
                multi_col_group[name] = col_group
        elif col_group in df_columns:
            cat_names.append(col_group)
    return cat_names, multi_col_group


def _is_list_col(column_group, df):
    has_lists = any(_is_list_dtype(df[col]) for col in column_group)
    if has_lists and len(column_group) != 1:
        raise ValueError("Can't categorical encode multiple list columns")
    return has_lists


def _hash_bucket(df, num_buckets, col, encode_type="joint"):
    if encode_type == "joint":
        nb = num_buckets[col[0]]
        encoded = _hash_series(df[col[0]]) % nb
    elif encode_type == "combo":
        if len(col) > 1:
            name = _make_name(*tuple(col), sep="_")
        else:
            name = col[0]
        nb = num_buckets[name]
        val = 0
        for column in col:
            val ^= _hash_series(df[column])  # or however we want to do this aggregation
        val = val % nb
        encoded = val
    return encoded


def _copy_storage(existing_stats, existing_path, new_path, copy):
    """ helper function to copy files to a new storage location"""
    from shutil import copyfile

    new_locations = {}
    for column, existing_file in existing_stats.items():
        new_file = existing_file.replace(str(existing_path), str(new_path))
        if copy and new_file != existing_file:
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            copyfile(existing_file, new_file)

        new_locations[column] = new_file

    return new_locations
