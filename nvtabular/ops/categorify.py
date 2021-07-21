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
from copy import deepcopy
from dataclasses import dataclass
from operator import getitem
from typing import Optional, Union

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
from pyarrow import parquet as pq

from nvtabular import dispatch
from nvtabular.dispatch import DataFrameType, annotate
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
        vocabs=None,
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
        if encode_type == "combo" and vocabs is not None:
            raise ValueError("Passing in vocabs is not supported with a combo encoding.")

        # Other self-explanatory initialization
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

        self.vocabs = {}
        if vocabs is not None:
            self.vocabs = self.process_vocabs(vocabs)
        self.categories = deepcopy(self.vocabs)

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
        cols_with_vocabs = list(self.categories.keys())
        columns = [
            list(c) if isinstance(c, tuple) else c for c in columns if c not in cols_with_vocabs
        ]
        dsk, key = _category_stats(ddf, self._create_fit_options_from_columns(columns))
        return Delayed(key, dsk)

    def fit_finalize(self, categories):
        for col in categories:
            self.categories[col] = categories[col]

    def clear(self):
        self.categories = deepcopy(self.vocabs)

    def process_vocabs(self, vocabs):
        categories = {}

        if dispatch._is_dataframe_object(vocabs):
            fit_options = self._create_fit_options_from_columns(list(vocabs.columns))
            base_path = os.path.join(self.out_path, fit_options.stat_name)
            os.makedirs(base_path, exist_ok=True)
            for col in list(vocabs.columns):
                col_df = vocabs[[col]]
                if col_df[col].iloc[0] is not None:
                    with_empty = dispatch._add_to_series(col_df[col], [None]).reset_index()[0]
                    vals = {col: with_empty}
                    col_df = dispatch._make_df(vals)

                save_path = os.path.join(base_path, f"unique.{col}.parquet")
                col_df.to_parquet(save_path)
                categories[col] = save_path
        elif isinstance(vocabs, dict) and all(isinstance(v, str) for v in vocabs.values()):
            categories = vocabs
        else:
            error = """Unrecognized vocab type,
            please provide either a dictionary with paths to a parquet files
            or a DataFrame that contains the vocabulary per column.
            """
            raise ValueError(error)

        return categories

    def _create_fit_options_from_columns(self, columns) -> "FitOptions":
        return FitOptions(
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
                    dtype=self.dtype,
                )
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

    def inference_initialize(self, columns, inference_config):
        # we don't currently support 'combo'
        if self.encode_type == "combo":
            warnings.warn("Falling back to unoptimized inference path for encode_type 'combo' ")
            return None
        import nvtabular_cpp

        return nvtabular_cpp.inference.CategorifyTransform(self)

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
    return cat_names


def get_embedding_sizes(source, output_dtypes=None):
    """Returns a dictionary of embedding sizes from a workflow or column_group

    Parameters
    ----------
    source : Workflow or ColumnGroup
        Either a nvtabular Workflow or ColumnGroup object that we should use to find
        embedding sizes
    output_dtypes : dict, optional
        Optional dictionary of column_name:dtype. If passing a workflow object dtypes
        will be read from the workflow. This is used to figure out which columns
        are multihot-categorical, which are split out by this function. If passed a column_group
        and this parameter isn't set, you won't have multihot columns returned separately
    """
    # TODO: do we need to distinguish multihot columns here?  (if so why? )

    # have to lazy import Workflow to avoid circular import errors
    from nvtabular.workflow import Workflow

    if isinstance(source, Workflow):
        queue = [source.column_group]
        output_dtypes = output_dtypes or source.output_dtypes
    else:
        # passed in a column group
        queue = [source]
        output_dtypes = output_dtypes or {}

    output = {}
    multihot_columns = set()
    while queue:
        current = queue.pop()
        if current.op and hasattr(current.op, "get_embedding_sizes"):
            output.update(current.op.get_embedding_sizes(current.columns))
        elif not current.op:
            # only follow parents if its not an operator node (which could
            # transform meaning of the get_embedding_sizes
            queue.extend(current.parents)

    for column in output:
        dtype = output_dtypes.get(column)
        if dtype and dispatch._is_list_dtype(dtype):
            # multi hot so remove from output and add to multihot
            multihot_columns.add(column)
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


@dataclass
class FitOptions:
    """Contains options on how to fit statistics.

    Parameters
    ----------
        col_groups: list
            Columns to group by
        agg_cols: list
            For groupby statistics, this is the list of continuous columns to calculate statistics
            for
        agg_list: list
            List of operations (sum/max/...) to perform on the grouped continuous columns
        out_path: str
            Where to write statistics in parquet format
        freq_limit: int or dict
            Categories with a count/frequency below this threshold will be
            ommited from the encoding and corresponding data will be mapped
            to the "null" category.
        tree_width:
           Tree width of the hash-based groupby reduction for each categorical column.
        on_host:
            Whether to convert cudf data to pandas between tasks in the groupby reduction.
        stat_name:
            Name of statistic to use when writing out statistics
        concat_groups:
            Whether to use a 'joint' vocabulary between columns
        name_sep:
            Delimiter to use for concatenating columns into a string
        max_size:
            The maximum size of an embedding table
        num_buckets:
            If specified will also do hashing operation for values that would otherwise be mapped
            to as unknown (by freq_limit or max_size parameters)
    """

    col_groups: list
    agg_cols: list
    agg_list: list
    out_path: str
    freq_limit: Union[int, dict]
    tree_width: Union[int, dict]
    on_host: bool
    stat_name: str = "categories"
    concat_groups: bool = False
    name_sep: str = "-"
    max_size: Optional[Union[int, dict]] = None
    num_buckets: Optional[Union[int, dict]] = None


@annotate("top_level_groupby", color="green", domain="nvt_python")
def _top_level_groupby(df, options: FitOptions):
    sum_sq = "std" in options.agg_list or "var" in options.agg_list
    calculate_min = "min" in options.agg_list
    calculate_max = "max" in options.agg_list

    # Top-level operation for category-based groupby aggregations
    output = {}
    k = 0
    for i, cat_col_group in enumerate(options.col_groups):
        if isinstance(cat_col_group, tuple):
            cat_col_group = list(cat_col_group)

        if isinstance(cat_col_group, str):
            cat_col_group = [cat_col_group]
        cat_col_group_str = _make_name(*cat_col_group, sep=options.name_sep)

        if options.concat_groups and len(cat_col_group) > 1:
            # Concatenate columns and replace cat_col_group
            # with the single name
            df_gb = type(df)()
            ignore_index = True
            df_gb[cat_col_group_str] = _concat([df[col] for col in cat_col_group], ignore_index)
            cat_col_group = [cat_col_group_str]
        else:
            # Compile aggregation dictionary and add "squared-sum"
            # column(s) (necessary when `agg_cols` is non-empty)
            df_gb = df[cat_col_group + options.agg_cols].copy(deep=False)

        agg_dict = {}
        agg_dict[cat_col_group[0]] = ["count"]
        for col in options.agg_cols:
            agg_dict[col] = ["sum"]
            if sum_sq:
                name = _make_name(col, "pow2", sep=options.name_sep)
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
            df_gb = dispatch._flatten_list_column(df_gb[cat_col_group[0]])

        # NOTE: groupby(..., dropna=False) requires pandas>=1.1.0
        gb = df_gb.groupby(cat_col_group, dropna=False).agg(agg_dict)
        gb.columns = [
            _make_name(*(tuple(cat_col_group) + name[1:]), sep=options.name_sep)
            if name[0] == cat_col_group[0]
            else _make_name(*(tuple(cat_col_group) + name), sep=options.name_sep)
            for name in gb.columns.to_flat_index()
        ]
        gb.reset_index(inplace=True, drop=False)
        del df_gb

        # Split the result by the hash value of the categorical column
        nsplits = options.tree_width[cat_col_group_str]
        for j, split in shuffle_group(
            gb, cat_col_group, 0, nsplits, nsplits, True, nsplits
        ).items():
            if options.on_host:
                output[k] = split.to_arrow(preserve_index=False)
            else:
                output[k] = split
            k += 1
        del gb
    return output


@annotate("mid_level_groupby", color="green", domain="nvt_python")
def _mid_level_groupby(dfs, col_group, freq_limit_val, options: FitOptions):
    if isinstance(col_group, str):
        col_group = [col_group]
    elif isinstance(col_group, tuple):
        col_group = list(col_group)

    if options.concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=options.name_sep)]

    if options.on_host:
        # Construct gpu DataFrame from pyarrow data.
        # `on_host=True` implies gpu-backed data.
        df = pa.concat_tables(dfs, promote=True)
        df = dispatch._from_host(df)
    else:
        df = _concat(dfs, ignore_index=True)
    groups = df.groupby(col_group, dropna=False)
    gb = groups.agg({col: _get_aggregation_type(col) for col in df.columns if col not in col_group})
    gb.reset_index(drop=False, inplace=True)

    name_count = _make_name(*(col_group + ["count"]), sep=options.name_sep)
    if options.freq_limit and not options.max_size:
        gb = gb[gb[name_count] >= freq_limit_val]

    required = col_group.copy()
    if "count" in options.agg_list:
        required.append(name_count)

    ddof = 1
    for cont_col in options.agg_cols:
        name_sum = _make_name(*(col_group + [cont_col, "sum"]), sep=options.name_sep)
        if "sum" in options.agg_list:
            required.append(name_sum)

        if "mean" in options.agg_list:
            name_mean = _make_name(*(col_group + [cont_col, "mean"]), sep=options.name_sep)
            required.append(name_mean)
            gb[name_mean] = gb[name_sum] / gb[name_count]

        if "min" in options.agg_list:
            name_min = _make_name(*(col_group + [cont_col, "min"]), sep=options.name_sep)
            required.append(name_min)

        if "max" in options.agg_list:
            name_max = _make_name(*(col_group + [cont_col, "max"]), sep=options.name_sep)
            required.append(name_max)

        if "var" in options.agg_list or "std" in options.agg_list:
            n = gb[name_count]
            x = gb[name_sum]
            x2 = gb[_make_name(*(col_group + [cont_col, "pow2", "sum"]), sep=options.name_sep)]
            result = x2 - x ** 2 / n
            div = n - ddof
            div[div < 1] = 1
            result /= div
            result[(n - ddof) == 0] = np.nan

            if "var" in options.agg_list:
                name_var = _make_name(*(col_group + [cont_col, "var"]), sep=options.name_sep)
                required.append(name_var)
                gb[name_var] = result
            if "std" in options.agg_list:
                name_std = _make_name(*(col_group + [cont_col, "std"]), sep=options.name_sep)
                required.append(name_std)
                gb[name_std] = np.sqrt(result)

    if options.on_host:
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
def _write_gb_stats(dfs, base_path, col_group, options: FitOptions):
    if options.concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=options.name_sep)]
    if isinstance(col_group, str):
        col_group = [col_group]

    rel_path = "cat_stats.%s.parquet" % (_make_name(*col_group, sep=options.name_sep))
    path = os.path.join(base_path, rel_path)
    pwriter = None
    if not options.on_host and len(dfs):
        # Want first non-empty df for schema (if there are any)
        _d = next((df for df in dfs if len(df)), dfs[0])
        pwriter = dispatch._parquet_writer_dispatch(_d, path=path, compression=None)

    # Loop over dfs and append to file
    # TODO: For high-cardinality columns, should support
    #       Dask-based to_parquet call here (but would need to
    #       support directory reading within dependent ops)
    n_writes = 0
    for df in dfs:
        if len(df):
            if options.on_host:
                # Use pyarrow - df is already a pyarrow table
                if pwriter is None:
                    pwriter = pq.ParquetWriter(path, df.schema, compression=None)
                pwriter.write_table(df)
            else:
                # df is a cudf or pandas DataFrame
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
def _write_uniques(dfs, base_path, col_group, options):
    if options.concat_groups and len(col_group) > 1:
        col_group = [_make_name(*col_group, sep=options.name_sep)]
    if isinstance(col_group, str):
        col_group = [col_group]
    if options.on_host:
        # Construct gpu DataFrame from pyarrow data.
        # `on_host=True` implies gpu-backed data.
        df = pa.concat_tables(dfs, promote=True)
        df = dispatch._from_host(df)
    else:
        df = _concat(dfs, ignore_index=True)
    rel_path = "unique.%s.parquet" % (_make_name(*col_group, sep=options.name_sep))
    path = "/".join([base_path, rel_path])
    if len(df):
        # Make sure first category is Null
        df = df.sort_values(col_group, na_position="first")
        new_cols = {}
        nulls_missing = False
        for col in col_group:
            name_count = col + "_count"
            if options.max_size:
                max_emb_size = options.max_size
                if isinstance(options.max_size, dict):
                    max_emb_size = max_emb_size[col]
                if options.num_buckets:
                    if isinstance(options.num_buckets, int):
                        nlargest = max_emb_size - options.num_buckets - 1
                    else:
                        nlargest = max_emb_size - options.num_buckets[col] - 1
                else:
                    nlargest = max_emb_size - 1

                if nlargest <= 0:
                    raise ValueError("`nlargest` cannot be 0 or negative")

                if nlargest < len(df):
                    df = df.nlargest(n=nlargest, columns=name_count)

            if not dispatch._series_has_nulls(df[col]):
                if name_count in df:
                    df = df.sort_values(name_count, ascending=False, ignore_index=True)

                nulls_missing = True
                new_cols[col] = _concat(
                    [df._constructor_sliced([None], dtype=df[col].dtype), df[col]],
                    ignore_index=True,
                )
            else:
                # ensure None aka "unknown" stays at index 0
                if name_count in df:
                    df_0 = df.iloc[0:1]
                    df_1 = df.iloc[1:].sort_values(name_count, ascending=False, ignore_index=True)
                    df = _concat([df_0, df_1])
                new_cols[col] = df[col].copy(deep=False)
            if name_count in df:
                new_cols[name_count] = df[name_count].copy(deep=False)
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


def _groupby_to_disk(ddf, write_func, options: FitOptions):
    if not options.col_groups:
        return {}

    if options.concat_groups:
        if options.agg_list and options.agg_list != ["count"]:
            raise ValueError("Cannot use concat_groups=True with aggregations other than count")
        if options.agg_cols:
            raise ValueError("Cannot aggregate continuous-column stats with concat_groups=True")

    # Update tree_width
    tw = {}
    for col in options.col_groups:
        col = [col] if isinstance(col, str) else col
        if isinstance(col, tuple):
            col = list(col)

        col_str = _make_name(*col, sep=options.name_sep)
        if options.tree_width is None:
            tw[col_str] = 8
        elif isinstance(options.tree_width, int):
            tw[col_str] = options.tree_width
        else:
            tw[col_str] = options.tree_width.get(col_str, None) or 8
    options.tree_width = tw

    # Make dedicated output directory for the categories
    fs = get_fs_token_paths(options.out_path)[0]
    out_path = fs.sep.join([options.out_path, options.stat_name])
    fs.mkdirs(out_path, exist_ok=True)

    dsk = {}
    token = tokenize(
        ddf,
        options.col_groups,
        options.out_path,
        options.freq_limit,
        options.tree_width,
        options.on_host,
    )
    level_1_name = "level_1-" + token
    split_name = "split-" + token
    level_2_name = "level_2-" + token
    level_3_name = "level_3-" + token
    finalize_labels_name = options.stat_name + "-" + token
    for p in range(ddf.npartitions):
        dsk[(level_1_name, p)] = (_top_level_groupby, (ddf._name, p), options)
        k = 0
        for c, col in enumerate(options.col_groups):
            col = [col] if isinstance(col, str) else col
            col_str = _make_name(*col, sep=options.name_sep)
            for s in range(options.tree_width[col_str]):
                dsk[(split_name, p, c, s)] = (getitem, (level_1_name, p), k)
                k += 1

    col_groups_str = []
    for c, col in enumerate(options.col_groups):
        col = [col] if isinstance(col, str) else col
        col_str = _make_name(*col, sep=options.name_sep)
        col_groups_str.append(col_str)
        freq_limit_val = None
        if options.freq_limit:
            freq_limit_val = (
                options.freq_limit[col_str]
                if isinstance(options.freq_limit, dict)
                else options.freq_limit
            )
        for s in range(options.tree_width[col_str]):
            dsk[(level_2_name, c, s)] = (
                _mid_level_groupby,
                [(split_name, p, c, s) for p in range(ddf.npartitions)],
                col,
                freq_limit_val,
                options,
            )

        dsk[(level_3_name, c)] = (
            write_func,
            [(level_2_name, c, s) for s in range(options.tree_width[col_str])],
            out_path,
            col,
            options,
        )

    dsk[finalize_labels_name] = (
        _finish_labels,
        [(level_3_name, c) for c, col in enumerate(options.col_groups)],
        col_groups_str,
    )
    graph = HighLevelGraph.from_collections(finalize_labels_name, dsk, dependencies=[ddf])
    return graph, finalize_labels_name


def _category_stats(ddf, options: FitOptions):
    # Check if we only need categories
    if options.agg_cols == [] and options.agg_list == []:
        options.agg_list = ["count"]
        return _groupby_to_disk(ddf, _write_uniques, options)

    # Otherwise, getting category-statistics
    if isinstance(options.agg_cols, str):
        options.agg_cols = [options.agg_cols]
    if options.agg_list == []:
        options.agg_list = ["count"]

    return _groupby_to_disk(ddf, _write_gb_stats, options)


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
    dtype=None,
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
        read_pq_func = dispatch._read_parquet_dispatch(df)
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
            value = read_pq_func(  # pylint: disable=unexpected-keyword-arg
                path, columns=selection_r
            )
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
            codes = dispatch._flatten_list_column(df[selection_l[0]])
            codes["order"] = dispatch._arange(len(codes), like_df=df)
        else:
            codes = type(df)({"order": dispatch._arange(len(df), like_df=df)}, index=df.index)
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
        labels = dispatch._encode_list_column(df[selection_l[0]], labels, dtype=dtype)
    elif dtype:
        labels = labels.astype(dtype, copy=False)

    return labels


def _read_groupby_stat_df(path, name, cat_cache, read_pq_func):
    if cat_cache is not None:
        cat_cache = cat_cache if isinstance(cat_cache, str) else cat_cache.get(name, "disk")
        with get_worker_cache("stats") as cache:
            if cache:
                return fetch_table_data(cache, path, cache=cat_cache, reader=read_pq_func)
    return read_pq_func(path)


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
    has_lists = any(dispatch._is_list_dtype(df[col]) for col in column_group)
    if has_lists and len(column_group) != 1:
        raise ValueError("Can't categorical encode multiple list columns")
    return has_lists


def _hash_bucket(df, num_buckets, col, encode_type="joint"):
    if encode_type == "joint":
        nb = num_buckets[col[0]]
        encoded = dispatch._hash_series(df[col[0]]) % nb
    elif encode_type == "combo":
        if len(col) > 1:
            name = _make_name(*tuple(col), sep="_")
        else:
            name = col[0]
        nb = num_buckets[name]
        val = 0
        for column in col:
            val ^= dispatch._hash_series(df[column])  # or however we want to do this aggregation
        val = val % nb
        encoded = val
    return encoded


def _copy_storage(existing_stats, existing_path, new_path, copy):
    """helper function to copy files to a new storage location"""
    from shutil import copyfile

    new_locations = {}
    for column, existing_file in existing_stats.items():
        new_file = existing_file.replace(str(existing_path), str(new_path))
        if copy and new_file != existing_file:
            os.makedirs(os.path.dirname(new_file), exist_ok=True)
            copyfile(existing_file, new_file)

        new_locations[column] = new_file

    return new_locations
