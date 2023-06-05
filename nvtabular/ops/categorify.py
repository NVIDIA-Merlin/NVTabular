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
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from operator import getitem
from pathlib import Path
from typing import Optional, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_ds
from dask.base import tokenize
from dask.blockwise import BlockIndex
from dask.core import flatten
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.core import _concat, new_dd_object
from dask.dataframe.shuffle import shuffle_group
from dask.delayed import Delayed
from dask.highlevelgraph import HighLevelGraph
from dask.utils import parse_bytes
from fsspec.core import get_fs_token_paths

from merlin.core import dispatch
from merlin.core.dispatch import DataFrameType, annotate, is_cpu_object, nullable_series
from merlin.core.utils import device_mem_size, run_on_worker
from merlin.dag.ops.stat_operator import StatOperator
from merlin.io.worker import fetch_table_data, get_worker_cache
from merlin.schema import Schema, Tags
from nvtabular.ops.operator import ColumnSelector, Operator

# Constants
# (NVTabular will reserve `0` for padding and `1` for nulls)
PAD_OFFSET = 0
NULL_OFFSET = 1
OOV_OFFSET = 2


class Categorify(StatOperator):
    """
    Most of the data set will contain categorical features,
    and these variables are typically stored as text values.
    Machine Learning algorithms don't support these text values.
    Categorify operation can be added to the workflow to
    transform categorical features into unique integer values.

    Encoding Convention::

        - `0`: Not used by `Categorify` (reserved for padding).
        - `1`: Null and NaN values.
        - `[2, 2 + num_buckets)`: OOV values (including hash buckets).
        - `[2 + num_buckets, max_size)`: Unique vocabulary.

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
        omitted from the encoding and corresponding data will be mapped
        to the OOV indices. Can be represented as both an integer or
        a dictionary with column names as keys and frequency limit as
        value. If dictionary is used, all columns targeted must be included
        in the dictionary.
    encode_type : {"joint", "combo"}, default "joint"
        If "joint", the columns within any multi-column group will be
        jointly encoded. If "combo", the combination of values will be
        encoded as a new column. Note that replacement is not allowed for
        "combo", because the same column name can be included in
        multiple groups.
    split_out : dict or int, optional
        Number of files needed to store the unique values of each categorical
        column. High-cardinality columns may require `split_out>1`, while
        low-cardinality columns should be fine with the `split_out=1` default.
        If passing a dict, each key and value should correspond to the column
        name and value, respectively. The default value is 1 for all columns.
    split_every : dict or int, optional
        Number of adjacent partitions to aggregate in each tree-reduction
        node. The default value is 8 for all columns.
    out_path : str, optional
        Root directory where groupby statistics will be written out in
        parquet format.
    on_host : bool, default True
        Whether to convert cudf data to pandas between tasks in the hash-based
        groupby reduction. The extra host <-> device data movement can reduce
        performance.  However, using `on_host=True` typically improves stability
        (by avoiding device-level memory pressure).
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
    num_buckets : int, or dictionary:{column: num_oov_indices}, optional
        Number of indices to reserve for out-of-vocabulary (OOV) encoding at
        transformation time. By default, all OOV values will be mapped to
        the same index (`2`). If `num_buckets` is set to an integer greater
        than one, a column-wise hash and modulo will be used to map each OOV
        value to an index in the range `[2, 2 + num_buckets)`. A dictionary
        may be used if the desired `num_buckets` behavior varies by column.
    max_size : int or dictionary:{column: max_size_value}, optional
        Set the maximum size of the expected embedding table for each column.
        For example, if `max_size` is set to 1000, only the first 997 most-
        frequent values will be included in the unique-value vocabulary, and
        all remaining non-null values will be mapped to the OOV indices
        (indices `0` and `1` will still be reserved for padding and nulls).
        To use multiple OOV indices for infrequent values, set the `num_buckets`
        parameter accordingly. Note that `max_size` cannot be combined with
        `freq_threshold`, and it cannot be less than `num_buckets + 2`. By
        default, the total number of encoding indices will be unconstrained.
    cardinality_memory_limit: int or str, optional
        Upper limit on the "allowed" memory usage of the internal DataFrame and Table objects
        used to store unique categories. By default, this limit is 12.5% of the total memory.
        Note that this argument is meant as a guide for internal optimizations and UserWarnings
        within NVTabular, and does not guarantee that the memory limit will be satisfied.
    """

    def __init__(
        self,
        freq_threshold=0,
        out_path=None,
        cat_cache="host",
        dtype=None,
        on_host=True,
        encode_type="joint",
        name_sep="_",
        search_sorted=False,
        num_buckets=None,
        vocabs=None,
        max_size=0,
        single_table=False,
        cardinality_memory_limit=None,
        tree_width=None,
        split_out=1,
        split_every=8,
        **kwargs,  # Deprecated/unsupported arguments
    ):
        # Handle deprecations and unsupported kwargs
        if "start_index" in kwargs:
            raise ValueError(
                "start_index is now deprecated. `Categorify` will always "
                "reserve index `0` for user-specific purposes, and will "
                "use index `1` for null values."
            )
        if "na_sentinel" in kwargs:
            raise ValueError(
                "na_sentinel is now deprecated. `Categorify` will always "
                "reserve index `1` for null values, and the following "
                "`num_buckets` indices for out-of-vocabulary values "
                "(or just index `2` if `num_buckets is None`)."
            )
        if kwargs:
            raise ValueError(f"Unrecognized key-word arguments: {kwargs}")

        # Warn user if they set num_buckets without setting max_size or
        # freq_threshold - This setting used to hash everything, but will
        # now just use multiple indices for OOV encodings at transform time
        if num_buckets and not (max_size or freq_threshold):
            warnings.warn(
                "You are setting num_buckets without using max_size or "
                "freq_threshold to restrict the number of distinct "
                "categories. Are you sure this is what you want?"
            )

        # We need to handle three types of encoding here:
        #
        #   (1) Conventional encoding. There are no multi-column groups. So,
        #       each categorical column is separately transformed into a new
        #       "encoded" column (1-to-1).  The unique values are calculated
        #       separately for each column.
        #
        #   (2) Multi-column "Joint" encoding (there are multi-column groups
        #       in `columns` and `encode_type="joint"`).  Still a
        #       1-to-1 transformation of categorical columns.  However,
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

        # Set workflow_nodes if the user has passed in a list of columns.
        # The purpose is to capture multi-column groups. If the user doesn't
        # specify `columns`, there are no multi-column groups to worry about.
        self.workflow_nodes = None
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
        self.single_table = single_table
        self.freq_threshold = freq_threshold or 0
        self.out_path = out_path or "./"
        self.dtype = dtype
        self.on_host = on_host
        self.cat_cache = cat_cache
        self.encode_type = encode_type
        self.search_sorted = search_sorted
        self.cardinality_memory_limit = cardinality_memory_limit
        self.split_every = split_every
        self.split_out = split_out
        _deprecate_tree_width(tree_width)

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
            # See: merlin.core.dispatch.hash_series
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
    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame):
        # User passed in a list of column groups. We need to figure out
        # if this list contains any multi-column groups, and if there
        # are any (obvious) problems with these groups
        columns_uniq = list(set(flatten(col_selector.names, container=tuple)))
        columns_all = list(flatten(col_selector.names, container=tuple))
        if sorted(columns_all) != sorted(columns_uniq) and self.encode_type == "joint":
            # If we are doing "joint" encoding, there must be unique mapping
            # between input column names and column groups.  Otherwise, more
            # than one unique-value table could be used to encode the same
            # column.
            raise ValueError("Same column name included in multiple groups.")

        for group in col_selector.subgroups:
            if len(group.names) > 1:
                # For multi-column groups, we concatenate column names
                # to get the "group" name.
                name = _make_name(*group.names, sep=self.name_sep)
                for col in group.names:
                    self.storage_name[col] = name

        # Check metadata type to reset on_host and cat_cache if the
        # underlying ddf is already a pandas-backed collection
        _cpu = False
        if isinstance(ddf._meta, pd.DataFrame):
            _cpu = True
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
            list(c) if isinstance(c, tuple) else c
            for c in col_selector.grouped_names
            if (_make_name(*c, sep=self.name_sep) if isinstance(c, tuple) else c)
            not in cols_with_vocabs
        ]
        if not columns:
            return Delayed("no-op", {"no-op": {}})

        # Define a rough row-count at which we are likely to
        # start hitting memory-pressure issues that cannot
        # be accommodated with smaller partition sizes.
        # By default, we estimate a "problematic" cardinality
        # to be one that consumes >12.5% of the total memory.
        self.cardinality_memory_limit = parse_bytes(
            self.cardinality_memory_limit or int(device_mem_size(kind="total", cpu=_cpu) * 0.125)
        )

        dsk, key = _category_stats(ddf, self._create_fit_options_from_columns(columns))
        return Delayed(key, dsk)

    def fit_finalize(self, categories):
        idx_count = 0
        for cat in categories:
            # this is a path
            self.categories[cat] = categories[cat]
            # check the argument
            if self.single_table:
                cat_file_path = self.categories[cat]
                idx_count, new_cat_file_path = run_on_worker(
                    _reset_df_index, cat, cat_file_path, idx_count
                )
                self.categories[cat] = new_cat_file_path

    def clear(self):
        """Clear the internal state of the operator's stats."""
        self.categories = deepcopy(self.vocabs)

    def process_vocabs(self, vocabs):
        """Process vocabs passed in by the user."""
        categories = {}
        if isinstance(vocabs, dict) and all(dispatch.is_series_object(v) for v in vocabs.values()):
            fit_options = self._create_fit_options_from_columns(list(vocabs.keys()))
            base_path = os.path.join(self.out_path, fit_options.stat_name)
            num_buckets = fit_options.num_buckets
            os.makedirs(base_path, exist_ok=True)
            for col, vocab in vocabs.items():
                col_name = _make_name(*col, sep=self.name_sep) if isinstance(col, tuple) else col
                vals = {col_name: vocab}
                oov_count = 1
                if num_buckets:
                    oov_count = (
                        num_buckets if isinstance(num_buckets, int) else num_buckets[col_name]
                    ) or 1
                col_df = dispatch.make_df(vals).dropna()
                col_df.index += NULL_OFFSET + oov_count
                save_path = _save_encodings(col_df, base_path, col_name)
                categories[col_name] = save_path
        elif isinstance(vocabs, dict) and all(isinstance(v, str) for v in vocabs.values()):
            # TODO: How to deal with the fact that this file may be missing null and oov rows??
            categories = {
                (_make_name(*col, sep=self.name_sep) if isinstance(col, tuple) else col): path
                for col, path in vocabs.items()
            }
        else:
            error = """Unrecognized vocab type,
            please provide either a dictionary with paths to parquet files
            or a dictionary with pandas Series objects.
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
            self.split_out,
            self.on_host,
            concat_groups=self.encode_type == "joint",
            name_sep=self.name_sep,
            max_size=self.max_size,
            num_buckets=self.num_buckets,
            cardinality_memory_limit=self.cardinality_memory_limit,
            split_every=self.split_every,
        )

    def set_storage_path(self, new_path, copy=False):
        self.categories = _copy_storage(self.categories, self.out_path, new_path, copy=copy)
        self.out_path = new_path

    @annotate("Categorify_transform", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        new_df = df.copy(deep=False)
        if isinstance(self.freq_threshold, dict):
            assert all(x in self.freq_threshold for x in col_selector.names)

        column_mapping = self.column_mapping(col_selector)
        column_names = list(column_mapping.keys())

        # Encode each column-group separately
        for name in column_names:
            try:
                # Use the column-group `list` directly (not the string name)
                use_name = column_mapping.get(name, name)

                # Storage name may be different than group for case (2)
                # Only use the "aliased" `storage_name` if we are dealing with
                # a multi-column group, or if we are doing joint encoding
                if isinstance(use_name, (list, tuple)) and len(use_name) == 1:
                    use_name = use_name[0]

                if isinstance(use_name, (list, tuple)) and len(use_name) == 1:
                    use_name = use_name[0]

                if use_name != name or self.encode_type == "joint":
                    storage_name = self.storage_name.get(name, name)
                else:
                    storage_name = name

                if isinstance(use_name, tuple):
                    use_name = list(use_name)

                path = self.categories[storage_name]

                encoded = _encode(
                    use_name,
                    storage_name,
                    path,
                    df,
                    self.cat_cache,
                    freq_threshold=self.freq_threshold[name]
                    if isinstance(self.freq_threshold, dict)
                    else self.freq_threshold,
                    search_sorted=self.search_sorted,
                    buckets=self.num_buckets,
                    encode_type=self.encode_type,
                    cat_names=column_names,
                    max_size=self.max_size,
                    dtype=self.output_dtype,
                    split_out=(
                        self.split_out.get(storage_name, 1)
                        if isinstance(self.split_out, dict)
                        else self.split_out
                    ),
                    single_table=self.single_table,
                )
                new_df[name] = encoded
            except Exception as e:
                raise RuntimeError(f"Failed to categorical encode column {name}") from e

        return new_df

    def column_mapping(self, col_selector):
        column_mapping = {}
        if self.encode_type == "combo":
            for group in col_selector.grouped_names:
                if isinstance(group, (tuple, list)):
                    name = _make_name(*group, sep=self.name_sep)
                    group = [*group]
                else:
                    name = group
                    group = [group]

                column_mapping[name] = group
        else:
            column_mapping = super().column_mapping(col_selector)
        return column_mapping

    def _compute_properties(self, col_schema, input_schema):
        new_schema = super()._compute_properties(col_schema, input_schema)
        col_name = col_schema.name

        category_name = self.storage_name.get(col_name, col_name)
        target_category_path = self.categories.get(category_name, None)

        cardinality, dimensions = self.get_embedding_sizes([category_name])[category_name]

        to_add = {
            "num_buckets": self.num_buckets[col_name]
            if isinstance(self.num_buckets, dict)
            else self.num_buckets,
            "freq_threshold": self.freq_threshold[col_name]
            if isinstance(self.freq_threshold, dict)
            else self.freq_threshold,
            "max_size": self.max_size[col_name]
            if isinstance(self.max_size, dict)
            else self.max_size,
            "cat_path": target_category_path,
            "domain": {"min": 0, "max": cardinality - 1, "name": category_name},
            "embedding_sizes": {"cardinality": cardinality, "dimension": dimensions},
        }

        return col_schema.with_properties({**new_schema.properties, **to_add})

    @property
    def output_tags(self):
        return [Tags.CATEGORICAL]

    @property
    def output_dtype(self):
        return self.dtype or np.int64

    def compute_selector(
        self,
        input_schema: Schema,
        selector: ColumnSelector,
        parents_selector: ColumnSelector,
        dependencies_selector: ColumnSelector,
    ) -> ColumnSelector:
        self._validate_matching_cols(input_schema, parents_selector, "computing input selector")
        return parents_selector

    def get_embedding_sizes(self, columns):
        return _get_embeddings_dask(self.categories, columns, self.num_buckets)

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


def get_embedding_sizes(source, output_dtypes=None):
    """Returns a dictionary of embedding sizes from a workflow or workflow_node

    Parameters
    ----------
    source : Workflow or ColumnSelector
        Either a nvtabular Workflow or ColumnSelector object that we should use to find
        embedding sizes
    output_dtypes : dict, optional
        Optional dictionary of column_name:dtype. If passing a workflow object dtypes
        will be read from the workflow. This is used to figure out which columns
        are multihot-categorical, which are split out by this function. If passed a workflow_node
        and this parameter isn't set, you won't have multihot columns returned separately
    """
    # TODO: do we need to distinguish multihot columns here?  (if so why? )

    # have to lazy import Workflow to avoid circular import errors
    from nvtabular.workflow import Workflow

    output_node = source.output_node if isinstance(source, Workflow) else source

    if isinstance(source, Workflow):
        output_dtypes = output_dtypes or source.output_dtypes
    else:
        # passed in a column group
        output_dtypes = output_dtypes or {}

    output = {}
    multihot_columns = set()
    cats_schema = output_node.output_schema.select_by_tag(Tags.CATEGORICAL)
    for col_name, col_schema in cats_schema.column_schemas.items():
        if col_schema.dtype and col_schema.is_list and col_schema.is_ragged:
            # multi hot so remove from output and add to multihot
            multihot_columns.add(col_name)

        embeddings_sizes = col_schema.properties.get("embedding_sizes", {})
        cardinality = embeddings_sizes["cardinality"]
        dimensions = embeddings_sizes["dimension"]
        output[col_name] = (cardinality, dimensions)

    # TODO: returning different return types like this (based off the presence
    # of multihot features) is pretty janky. fix.
    if not multihot_columns:
        return output

    single_hots = {k: v for k, v in output.items() if k not in multihot_columns}
    multi_hots = {k: v for k, v in output.items() if k in multihot_columns}
    return single_hots, multi_hots


def _get_embeddings_dask(paths, cat_names, buckets=0):
    embeddings = {}
    if isinstance(buckets, int):
        buckets = {name: buckets for name in cat_names}
    for col in cat_names:
        path = paths.get(col)
        num_rows = OOV_OFFSET
        if path:
            for file_frag in pa_ds.dataset(path, format="parquet").get_fragments():
                num_rows += file_frag.metadata.num_rows
        if isinstance(buckets, dict):
            bucket_size = buckets.get(col, 0)
        elif isinstance(buckets, int):
            bucket_size = buckets
        else:
            bucket_size = 1
        num_rows += bucket_size
        embeddings[col] = _emb_sz_rule(num_rows)
    return embeddings


def _emb_sz_rule(n_cat: int, minimum_size=16, maximum_size=512) -> int:
    return n_cat, min(max(minimum_size, round(1.6 * n_cat**0.56)), maximum_size)


def _make_name(*args, sep="_"):
    return sep.join(args)


def _to_parquet_dask_lazy(df, path, write_index=False):
    # Write DataFrame data to parquet (lazily) with dask

    # Check if we already have a dask collection
    is_collection = isinstance(df, DaskDataFrame)

    # Use `ddf.to_parquet` method
    kwargs = {
        "overwrite": True,
        "compute": False,
        "write_index": write_index,
        "schema": None,
    }
    return (
        df
        if is_collection
        else dispatch.convert_data(
            df,
            cpu=isinstance(df, pd.DataFrame),
            to_collection=True,
        )
    ).to_parquet(path, **kwargs)


def _save_encodings(
    df,
    base_path,
    field_name,
    preserve_index=False,
    first_n=None,
    freq_threshold=None,
    oov_count=1,
    null_size=None,
):
    # Write DataFrame data to parquet (eagerly) with dask

    # Define paths
    unique_path = "/".join([str(base_path), f"unique.{field_name}.parquet"])
    meta_path = "/".join([str(base_path), f"meta.{field_name}.parquet"])

    # Check if we already have a dask collection
    is_collection = isinstance(df, DaskDataFrame)

    # Create empty directory if it doesn't already exist
    use_directory = is_collection and df.npartitions > 1
    fs = get_fs_token_paths(unique_path, mode="wb")[0]
    _path = fs._strip_protocol(unique_path)
    if fs.isdir(_path) or fs.exists(_path):
        fs.rm(_path, recursive=True)
    if use_directory:
        fs.mkdir(_path, exists_ok=True)

    # Start tracking embedding metadata
    record_size_meta = True
    oov_size = 0
    unique_count = 0
    unique_size = 0

    # Iterate over partitions and write to disk
    size = oov_count + OOV_OFFSET  # Reserve null and oov buckets
    for p, part in enumerate(df.partitions if is_collection else [df]):
        local_path = "/".join([unique_path, f"part.{p}.parquet"]) if use_directory else unique_path
        _df = _compute_sync(part) if is_collection else part
        _len = len(_df)
        if _len == 0:
            continue

        size_col = f"{field_name}_size"
        if size_col not in _df.columns:
            record_size_meta = False

        if record_size_meta:
            # Set number of rows allowed from this part
            if first_n is not None:
                first_n_local = first_n - size
            else:
                first_n_local = _len

            # Update oov size
            if first_n or freq_threshold:
                removed = None
                if freq_threshold:
                    sizes = _df[size_col]
                    removed = df[(sizes < freq_threshold) & (sizes > 0)]
                    _df = _df[(sizes >= freq_threshold) | (sizes == 0)]
                if first_n and _len > first_n_local:
                    removed = _df.iloc[first_n_local:]
                    _df = _df.iloc[:first_n_local]
                if removed is not None:
                    oov_size += removed[size_col].sum()
                    _len = len(_df)

            # Record unique-value metadata
            unique_size += _df[size_col].sum()

        if not preserve_index:
            # If we are NOT writing the index of df,
            # then make sure we are writing a "correct"
            # index. Note that we avoid using ddf.to_parquet
            # so that we can make sure the index is correct
            _df.set_index(
                pd.RangeIndex(
                    start=size,
                    stop=size + _len,
                    step=1,
                ),
                drop=True,
                inplace=True,
            )

        size += _len
        unique_count += _len
        _df.to_parquet(local_path, compression=None)
        if first_n and size >= first_n:
            break  # Ignore any remaining files

    # Write encoding metadata
    meta = {
        "kind": ["pad", "null", "oov", "unique"],
        "offset": [PAD_OFFSET, NULL_OFFSET, OOV_OFFSET, OOV_OFFSET + oov_count],
        "num_indices": [1, 1, oov_count, unique_count],
    }
    if record_size_meta:
        meta["num_observed"] = [0, null_size, oov_size, unique_size]
    type(_df)(meta).to_parquet(meta_path)

    # Return path to uniques
    return unique_path


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
            omitted from the encoding and corresponding data will be mapped
            to the "null" category.
        split_out:
           Number of output partitions to use for each category in ``fit``.
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
        cardinality_memory_limit: int
            Suggested upper limit on categorical data containers.
        split_every:
            Number of adjacent partitions to reduce in each tree node.
    """

    col_groups: list
    agg_cols: list
    agg_list: list
    out_path: str
    freq_limit: Union[int, dict]
    split_out: Union[int, dict]
    on_host: bool
    stat_name: str = "categories"
    concat_groups: bool = False
    name_sep: str = "-"
    max_size: Optional[Union[int, dict]] = None
    num_buckets: Optional[Union[int, dict]] = None
    cardinality_memory_limit: Optional[int] = None
    split_every: Optional[Union[int, dict]] = 8

    def __post_init__(self):
        if not isinstance(self.col_groups, ColumnSelector):
            self.col_groups = ColumnSelector(self.col_groups)

        col_selectors = []
        for cat_col_names in self.col_groups.grouped_names:
            if isinstance(cat_col_names, tuple):
                cat_col_names = list(cat_col_names)

            if isinstance(cat_col_names, str):
                cat_col_names = [cat_col_names]

            if not isinstance(cat_col_names, ColumnSelector):
                cat_col_selector = ColumnSelector(cat_col_names)
            else:
                cat_col_selector = cat_col_names

            col_selectors.append(cat_col_selector)

        self.col_groups = col_selectors


def _general_concat(
    frames,
    cardinality_memory_limit=False,
    col_selector=None,
    **kwargs,
):
    # Concatenate DataFrame or pa.Table objects
    if isinstance(frames[0], pa.Table):
        df = pa.concat_tables(frames, promote=True)
        if (
            cardinality_memory_limit
            and col_selector is not None
            and df.nbytes > cardinality_memory_limit
        ):
            # Before fully converting this pyarrow Table
            # to a cudf DatFrame, we can reduce the memory
            # footprint of `df`. Since the size of `df`
            # depends on the cardinality of the features,
            # and NOT on the partition size, the remaining
            # logic in this function has an OOM-error risk
            # (even with tiny partitions).
            size_columns = []
            for col in col_selector.names:
                name = col + "_size"
                if name in df.schema.names:
                    # Convert this column alone to cudf,
                    # and drop the field from df. Note that
                    # we are only converting this column to
                    # cudf to take advantage of fast `max`
                    # performance.
                    size_columns.append(dispatch.from_host(df.select([name])))
                    df = df.drop([name])
                    # Use numpy to calculate the "minimum"
                    # dtype needed to capture the "size" column,
                    # and cast the type
                    typ = np.min_scalar_type(size_columns[-1][name].max() * 2)
                    size_columns[-1][name] = size_columns[-1][name].astype(typ)
            # Convert the remaining columns in df to cudf,
            # and append the type-casted "size" columns
            df = dispatch.concat_columns([dispatch.from_host(df)] + size_columns)
        else:
            # Empty DataFrame - No need for type-casting
            df = dispatch.from_host(df)
        return df
    else:
        # For now, if we are not concatenating in host memory,
        # we will assume that reducing the memory footprint of
        # "size" columns is not a priority. However, the same
        # type-casting optimization can also be done for both
        # pandas and cudf-backed data here.
        return _concat(frames, **kwargs)


@annotate("top_level_groupby", color="green", domain="nvt_python")
def _top_level_groupby(df, options: FitOptions = None, spill=True):
    assert options is not None
    sum_sq = "std" in options.agg_list or "var" in options.agg_list
    calculate_min = "min" in options.agg_list
    calculate_max = "max" in options.agg_list
    # Top-level operation for category-based groupby aggregations
    output = {}
    k = 0
    for i, cat_col_names in enumerate(options.col_groups):
        if not isinstance(cat_col_names, ColumnSelector):
            cat_col_selector = ColumnSelector(cat_col_names)
        else:
            cat_col_selector = cat_col_names

        cat_col_selector_str = _make_name(*cat_col_selector.names, sep=options.name_sep)

        if options.concat_groups and len(cat_col_selector.names) > 1:
            # Concatenate columns and replace cat_col_group
            # with the single name
            df_gb = type(df)()
            ignore_index = True
            df_gb[cat_col_selector_str] = _concat(
                [_maybe_flatten_list_column(col, df)[col] for col in cat_col_selector.names],
                ignore_index,
            )
            cat_col_selector = ColumnSelector([cat_col_selector_str])
        else:
            # Compile aggregation dictionary and add "squared-sum"
            # column(s) (necessary when `agg_cols` is non-empty)
            combined_col_selector = cat_col_selector + options.agg_cols

            df_gb = df[combined_col_selector.names].copy(deep=False)

        agg_dict = {}
        base_aggs = []
        if "size" in options.agg_list:
            # This is either for a Categorify operation,
            # or "size" is in the list of aggregations
            base_aggs.append("size")
        if set(options.agg_list).difference({"size", "min", "max"}):
            # This is a groupby aggregation that may
            # require "count" statistics
            base_aggs.append("count")
        agg_dict[cat_col_selector.names[0]] = base_aggs
        if isinstance(options.agg_cols, list):
            options.agg_cols = ColumnSelector(options.agg_cols)
        for col in options.agg_cols.names:
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
        df_gb = _maybe_flatten_list_column(cat_col_selector.names[0], df_gb)
        # NOTE: groupby(..., dropna=False) requires pandas>=1.1.0
        gb = df_gb.groupby(cat_col_selector.names, dropna=False).agg(agg_dict)
        gb.columns = [
            _make_name(*(tuple(cat_col_selector.names) + name[1:]), sep=options.name_sep)
            if name[0] == cat_col_selector.names[0]
            else _make_name(*(tuple(cat_col_selector.names) + name), sep=options.name_sep)
            for name in gb.columns.to_flat_index()
        ]
        gb.reset_index(inplace=True, drop=False)
        del df_gb

        # Extract null groups into gb_null
        isnull = gb.isnull().any(axis=1)
        gb_null = gb[~isnull]
        gb = gb[isnull]
        if not len(gb_null):
            gb_null = None
        del isnull

        # Split the result by the hash value of the categorical column
        nsplits = options.split_out[cat_col_selector_str]
        for j, split in shuffle_group(
            gb, cat_col_selector.names, 0, nsplits, nsplits, True, nsplits
        ).items():
            if gb_null is not None:
                # Guarantee that the first split will contain null groups
                split = _concat([gb_null, split], ignore_index=True)
                gb_null = None
            if spill and options.on_host and not is_cpu_object(split):
                output[k] = split.to_arrow(preserve_index=False)
            else:
                output[k] = split
            k += 1
        del gb
    return output


@annotate("mid_level_groupby", color="green", domain="nvt_python")
def _mid_level_groupby(dfs, col_selector: ColumnSelector, options: FitOptions, spill=True):
    if options.concat_groups and len(col_selector.names) > 1:
        col_selector = ColumnSelector([_make_name(*col_selector.names, sep=options.name_sep)])

    df = _general_concat(dfs, ignore_index=True)
    groups = df.groupby(col_selector.names, dropna=False)
    gb = groups.agg(
        {col: _get_aggregation_type(col) for col in df.columns if col not in col_selector.names}
    )
    gb.reset_index(drop=False, inplace=True)

    if spill and options.on_host and not is_cpu_object(gb):
        gb_pd = gb.to_arrow(preserve_index=False)
        del gb
        return gb_pd
    return gb


@annotate("bottom_level_groupby", color="green", domain="nvt_python")
def _bottom_level_groupby(dfs, col_selector: ColumnSelector, options: FitOptions, spill=True):
    gb = _mid_level_groupby(dfs, col_selector, options, spill=False)
    if options.concat_groups and len(col_selector.names) > 1:
        col_selector = ColumnSelector([_make_name(*col_selector.names, sep=options.name_sep)])

    name_count = _make_name(*(col_selector.names + ["count"]), sep=options.name_sep)
    name_size = _make_name(*(col_selector.names + ["size"]), sep=options.name_sep)

    required = col_selector.names.copy()
    if "count" in options.agg_list:
        required.append(name_count)
    if "size" in options.agg_list:
        required.append(name_size)
    ddof = 1
    if isinstance(options.agg_cols, list):
        options.agg_cols = ColumnSelector(options.agg_cols)
    for cont_col in options.agg_cols.names:
        name_sum = _make_name(*(col_selector.names + [cont_col, "sum"]), sep=options.name_sep)
        if "sum" in options.agg_list:
            required.append(name_sum)

        if "mean" in options.agg_list:
            name_mean = _make_name(*(col_selector.names + [cont_col, "mean"]), sep=options.name_sep)
            required.append(name_mean)
            gb[name_mean] = gb[name_sum] / gb[name_count]

        if "min" in options.agg_list:
            name_min = _make_name(*(col_selector.names + [cont_col, "min"]), sep=options.name_sep)
            required.append(name_min)

        if "max" in options.agg_list:
            name_max = _make_name(*(col_selector.names + [cont_col, "max"]), sep=options.name_sep)
            required.append(name_max)

        if "var" in options.agg_list or "std" in options.agg_list:
            n = gb[name_count]
            x = gb[name_sum]
            x2 = gb[
                _make_name(*(col_selector.names + [cont_col, "pow2", "sum"]), sep=options.name_sep)
            ]
            result = x2 - x**2 / n
            div = n - ddof
            div[div < 1] = 1
            result /= div
            result[(n - ddof) == 0] = np.nan

            if "var" in options.agg_list:
                name_var = _make_name(
                    *(col_selector.names + [cont_col, "var"]), sep=options.name_sep
                )
                required.append(name_var)
                gb[name_var] = result
            if "std" in options.agg_list:
                name_std = _make_name(
                    *(col_selector.names + [cont_col, "std"]), sep=options.name_sep
                )
                required.append(name_std)
                gb[name_std] = np.sqrt(result)

    if spill and options.on_host and not is_cpu_object(gb[required]):
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


@annotate("write_uniques", color="green", domain="nvt_python")
def _write_uniques(
    dfs,
    base_path,
    col_selector: ColumnSelector,
    options: FitOptions,
    cpu: bool,
    path: str = None,
):
    """Writes out a dataframe to a parquet file.

    Parameters
    ----------
    dfs : DataFrame
    base_path : str
    col_selector :
    options : FitOptions

    Raises
    ------
    ValueError
        If the computed nlargest value is non-positive.

    Returns
    -------
    path : str
        the path to the output parquet file.

    """
    if options.concat_groups and len(col_selector.names) > 1:
        col_selector = ColumnSelector([_make_name(*col_selector.names, sep=options.name_sep)])

    # Set max_emb_size
    # This is the maximum number of rows we will write to
    # the unique-value parquet files
    col_name = col_selector.names[0]
    max_emb_size = options.max_size
    if max_emb_size:
        max_emb_size = max_emb_size[col_name] if isinstance(max_emb_size, dict) else max_emb_size

    # Set num_buckets
    # This is the maximum number of indices
    num_buckets = options.num_buckets
    if num_buckets:
        num_buckets = num_buckets if isinstance(num_buckets, int) else num_buckets[col_name]
    oov_count = num_buckets or 1

    # Set freq_threshold
    # This is the minimum unique count for a distinct
    # category to be included in the unique-value files
    freq_threshold = options.freq_limit
    if freq_threshold:
        freq_threshold = (
            freq_threshold if isinstance(freq_threshold, int) else freq_threshold[col_name]
        )

    # Sanity check
    if max_emb_size and max_emb_size < oov_count + 2:
        raise ValueError(
            "`max_size` can never be less than the maximum of "
            "`num_buckets + 2` and `3`, because we must always "
            "reserve pad, null and at least 1 oov-bucket index."
        )

    null_size = None
    if path:
        # We have a parquet path to construct uniques from
        # (rather than a list of DataFrame objects)
        df = dispatch.read_dispatch(cpu=cpu, collection=True)(
            path,
            split_row_groups=False,
        ).reset_index(drop=True)

        # Check if we need to compute the DataFrame collection
        # of unique values. For now, we can avoid doing this when
        # we are not jointly encoding multiple columns
        if simple := (len(col_selector.names) == 1 and df.npartitions > 1):
            col_name = col_selector.names[0]
            name_size = col_name + "_size"
            has_size = name_size in df
            try:
                # Sort by col_name
                df = df.sort_values(col_name, na_position="first")
            except (NotImplementedError, TypeError):
                # Dask-based sort failed - Need to compute first
                simple = False

        # At this point, `simple` may have changed from True to False
        # if the backend library failed to sort by the target column.
        if simple:
            # Define the null row
            def _drop_first_row(part, index):
                return part.iloc[1:] if index == (0,) else part

            null_row = df.head(1)
            if null_row[col_name].iloc[:1].isnull().any():
                df = df.map_partitions(_drop_first_row, BlockIndex((df.npartitions,)))
                if has_size:
                    null_size = null_row[name_size].iloc[0]
            else:
                null_size = 0

            # Sort by size (without null and oov rows)
            if has_size:
                # Avoid using dask_cudf to calculate divisions
                # (since it may produce too-few partitions)
                df = df.sort_values(
                    name_size,
                    ascending=False,
                    divisions=dd.shuffle._calculate_divisions(
                        df, df[name_size], False, df.npartitions
                    )[0][::-1],
                )

            unique_path = _save_encodings(
                df,
                base_path,
                _make_name(*col_selector.names, sep=options.name_sep),
                first_n=max_emb_size,
                freq_threshold=freq_threshold,
                oov_count=oov_count,
                null_size=null_size,
            )

            # TODO: Delete temporary parquet file(s) now thet the final
            # uniques are written to disk? (May not want to wait on deletion)
            return unique_path

        # If we have reached this point, we have a dask collection
        # that must be computed before continuing
        df = _compute_sync(df)
    else:
        # We have a list of DataFrame objects.
        # Collect aggregation results into single frame
        df = _general_concat(
            dfs,
            cardinality_memory_limit=options.cardinality_memory_limit,
            col_selector=col_selector,
            ignore_index=True,
        )

    # Check if we should warn user that this Column is likely
    # to cause memory-pressure issues
    _df_size = df.memory_usage(deep=True, index=True).sum()
    if (_df_size > options.cardinality_memory_limit) if options.cardinality_memory_limit else False:
        warnings.warn(
            f"Category DataFrame (with columns: {df.columns}) is {_df_size} "
            f"bytes in size. This is large compared to the suggested "
            f"upper limit of {options.cardinality_memory_limit} bytes!"
            f"(12.5% of the total memory by default)"
        )

    if len(df):
        # Make sure first category is Null.
        # Use ignore_index=True to avoid allocating memory for
        # an index we don't even need
        df = df.sort_values(col_selector.names, na_position="first", ignore_index=True)
        name_size_multi = "_".join(col_selector.names + ["size"])
        has_size = name_size_multi in df

        # Check if we already have a null row
        has_nans = df[col_selector.names].iloc[0].transpose().isnull().all()
        if hasattr(has_nans, "iloc"):
            has_nans = has_nans[0]

        if has_nans:
            if has_size:
                null_size = df[name_size_multi].iloc[0]
            df = df.iloc[1:]
        else:
            null_size = 0
        if has_size:
            df = df.sort_values(name_size_multi, ascending=False, ignore_index=True)
        df_write = df
    else:
        if hasattr(df, "convert_dtypes"):
            df = df.convert_dtypes()
        df_null = type(df)({c: [None] for c in col_selector.names})
        for c in col_selector.names:
            df_null[c] = df_null[c].astype(df[c].dtype)
        df_write = df_null

    unique_path = _save_encodings(
        df_write,
        base_path,
        _make_name(*col_selector.names, sep=options.name_sep),
        first_n=max_emb_size,
        freq_threshold=freq_threshold,
        oov_count=oov_count,
        null_size=null_size,
    )
    del df
    del df_write
    return unique_path


def _finish_labels(paths, cols):
    return {col: paths[i] for i, col in enumerate(cols)}


def _groupby_to_disk(ddf, write_func, options: FitOptions):
    if not options.col_groups:
        raise ValueError("no column groups to aggregate")

    if options.concat_groups:
        if options.agg_list and not set(options.agg_list).issubset({"count", "size"}):
            raise ValueError(
                "Cannot use concat_groups=True with aggregations other than count and size"
            )
        if options.agg_cols:
            raise ValueError("Cannot aggregate continuous-column stats with concat_groups=True")

    # Update split_out and split_every
    so, se = {}, {}
    for col in options.col_groups:
        col = [col] if isinstance(col, str) else col
        if isinstance(col, tuple):
            col = list(col)
        col_str = _make_name(*col.names, sep=options.name_sep)

        for _d, _opt, _default in [
            (so, options.split_out, 1),
            (se, options.split_every, 8),
        ]:
            if _opt is None:
                _d[col_str] = _default
            elif isinstance(_opt, int):
                _d[col_str] = _opt
            else:
                _d[col_str] = _opt.get(col_str, _default)

    options.split_out = so
    options.split_every = se

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
        options.split_out,
        options.split_every,
        options.on_host,
    )
    split_name = "split-" + token
    reduce_1_name = "reduce_1-" + token
    reduce_3_name = "reduce_3-" + token
    finalize_labels_name = options.stat_name + "-" + token

    # Use map_partitions to improve task fusion
    grouped = ddf.to_bag(format="frame").map_partitions(
        _top_level_groupby, options=options, token="level_1"
    )
    _grouped_meta = _top_level_groupby(ddf._meta, options=options)
    _grouped_meta_col = {}

    dsk_split = defaultdict(dict)
    for p in range(ddf.npartitions):
        k = 0
        for c, col in enumerate(options.col_groups):
            col = [col] if isinstance(col, str) else col
            col_str = _make_name(*col.names, sep=options.name_sep)
            _grouped_meta_col[c] = _grouped_meta[k]
            for s in range(options.split_out[col_str]):
                dsk_split[c][(split_name, p, c, s)] = (getitem, (grouped.name, p), k)
                k += 1

    col_groups_str = []
    col_group_frames = []
    for c, col in enumerate(options.col_groups):
        col = [col] if isinstance(col, str) else col
        col_str = _make_name(*col.names, sep=options.name_sep)
        col_groups_str.append(col_str)
        reduce_2_name = f"reduce_2-{c}-" + token
        for s in range(options.split_out[col_str]):
            split_every = options.split_every[col_str]
            parts = ddf.npartitions
            widths = [parts]
            while parts > 1:
                parts = math.ceil(parts / split_every)
                widths.append(int(parts))
            height = len(widths)
            if height >= 2:
                # Loop over reduction levels
                for depth in range(1, height):
                    # Loop over reduction groups
                    for group in range(widths[depth]):
                        # Calculate inputs for the current group
                        p_max = widths[depth - 1]
                        lstart = split_every * group
                        lstop = min(lstart + split_every, p_max)
                        if depth == 1:
                            # Input nodes are from input layer
                            input_keys = [(split_name, p, c, s) for p in range(lstart, lstop)]
                        else:
                            # Input nodes are tree-reduction nodes
                            input_keys = [
                                (reduce_1_name, p, c, s, depth - 1) for p in range(lstart, lstop)
                            ]

                        # Define task
                        if depth == height - 1:
                            # Final Node
                            assert (
                                group == 0
                            ), f"group = {group}, not 0 for final tree reduction task"
                            dsk_split[c][(reduce_2_name, s)] = (
                                _bottom_level_groupby,
                                input_keys,
                                col,
                                options,
                                False,
                            )
                        else:
                            # Intermediate Node
                            dsk_split[c][(reduce_1_name, group, c, s, depth)] = (
                                _mid_level_groupby,
                                input_keys,
                                col,
                                options,
                            )
            else:
                # Deal with single-partition case
                dsk_split[c][(reduce_2_name, s)] = (
                    _bottom_level_groupby,
                    [(split_name, 0, c, s)],
                    col,
                    options,
                    False,
                )

        # Make DataFrame collection for column-group result
        _meta = _bottom_level_groupby(
            [_grouped_meta_col[c]],
            col,
            options,
            spill=False,
        )
        _divisions = (None,) * (options.split_out[col_str] + 1)
        graph = HighLevelGraph.from_collections(reduce_2_name, dsk_split[c], dependencies=[grouped])
        col_group_frames.append(new_dd_object(graph, reduce_2_name, _meta, _divisions))

        # Write data to (possibly temporary) parquet files
        cpu = isinstance(col_group_frames[-1]._meta, pd.DataFrame)
        if write_func is None:
            # Write results directly to disk, and use
            # a final "barrier" task
            if options.concat_groups and len(col) > 1:
                col_selector = ColumnSelector([_make_name(*col.names, sep=options.name_sep)])
            else:
                col_selector = col
            rel_path = "cat_stats.%s.parquet" % (
                _make_name(*col_selector.names, sep=options.name_sep)
            )
            path = os.path.join(out_path, rel_path)
            col_group_frames[-1] = _to_parquet_dask_lazy(col_group_frames[-1], path)
            # Barrier-only task
            dsk[(reduce_3_name, c)] = (
                lambda keys, path: path,
                col_group_frames[-1].__dask_keys__(),
                path,
            )
        else:
            # Possibly write data to temporary parquet files,
            # and perform write operation(s) in final `write_func` task
            assert callable(write_func)
            if col_group_frames[-1].npartitions > 1 and write_func.__name__ == "_write_uniques":
                path = os.path.join(out_path, f"tmp.uniques.{col_str}")
                col_group_frames[-1] = _to_parquet_dask_lazy(col_group_frames[-1], path)
            else:
                path = None
            # Write + barrier task
            dsk[(reduce_3_name, c)] = (
                write_func,
                col_group_frames[-1].__dask_keys__(),
                out_path,
                col,
                options,
                cpu,
                path,
            )

    # Tie everything together into a graph with a single output key
    dsk[finalize_labels_name] = (
        _finish_labels,
        [(reduce_3_name, c) for c, col in enumerate(options.col_groups)],
        col_groups_str,
    )
    graph = HighLevelGraph.from_collections(
        finalize_labels_name, dsk, dependencies=col_group_frames
    )
    return graph, finalize_labels_name


def _category_stats(ddf, options: FitOptions):
    # Check if we only need categories
    if options.agg_cols == [] and options.agg_list == []:
        options.agg_list = ["size"]
        return _groupby_to_disk(ddf, _write_uniques, options)

    # Otherwise, getting category-statistics
    if isinstance(options.agg_cols, str):
        options.agg_cols = [options.agg_cols]
    if options.agg_list == []:
        options.agg_list = ["count"]

    return _groupby_to_disk(ddf, None, options)


def _encode(
    name,
    storage_name,
    path,
    df,
    cat_cache,
    freq_threshold=0,
    search_sorted=False,
    buckets=None,
    encode_type="joint",
    cat_names=None,
    max_size=0,
    dtype=None,
    split_out=1,
    single_table=False,
):
    """The _encode method is responsible for transforming a dataframe by taking the written
    out vocabulary file and looking up values to translate inputs to numeric
    outputs.

    Parameters
    ----------
    name :
    storage_name : dict
    path : str
    df : DataFrame
    cat_cache :
    freq_threshold :  int
        Categories with a count or frequency below this threshold will
        be omitted from the encoding and corresponding data will be
        mapped to the "Null" category. Defaults to 0.
    search_sorted :
        Defaults to False.
    buckets :
        Defaults to None.
    encode_type :
        Defaults to "joint".
    cat_names :
        Defaults to None.
    max_size :
        Defaults to 0.
    dtype :
        Defaults to None.

    Returns
    -------
    labels : numpy ndarray or Pandas Series

    """
    if isinstance(buckets, int):
        buckets = {name: buckets for name in cat_names}
    value = None
    selection_l = ColumnSelector(name if isinstance(name, list) else [name])
    selection_r = ColumnSelector(name if isinstance(name, list) else [storage_name])
    list_col = is_list_col(selection_l, df)

    # Find number of oov buckets
    if buckets and storage_name in buckets:
        num_oov_buckets = buckets[storage_name]
        search_sorted = False
    else:
        num_oov_buckets = 1

    if path:
        read_pq_func = dispatch.read_dispatch(
            df,
            fmt="parquet",
            collection=split_out > 1,
        )
        if cat_cache is not None and split_out == 1:
            cat_cache = (
                cat_cache if isinstance(cat_cache, str) else cat_cache.get(storage_name, "disk")
            )
            if len(df):
                with get_worker_cache("cats") as cache:
                    value = fetch_table_data(
                        cache,
                        path,
                        columns=selection_r.names,
                        cache=cat_cache,
                        cats_only=True,
                        reader=read_pq_func,
                    )
                    if len(value) and value["labels"].iloc[0] < OOV_OFFSET + num_oov_buckets:
                        # See: https://github.com/rapidsai/cudf/issues/12837
                        value["labels"] += OOV_OFFSET + num_oov_buckets
        else:
            value = read_pq_func(  # pylint: disable=unexpected-keyword-arg
                path,
                columns=selection_r.names,
                **({"split_row_groups": False} if split_out > 1 else {}),
            )

            value.index = value.index.rename("labels")
            if split_out > 1:
                value = value.reset_index(drop=False)
                if type(df).__module__.split(".")[0] == "cudf":
                    # `cudf.read_parquet` may drop the RangeIndex, so we need
                    # to use the parquet metadata to set a proper RangeIndex.
                    # We can avoid this workaround for cudf>=23.04
                    # (See: https://github.com/rapidsai/cudf/issues/12837)
                    ranges, size = [], OOV_OFFSET + num_oov_buckets
                    for file_frag in pa_ds.dataset(path, format="parquet").get_fragments():
                        part_size = file_frag.metadata.num_rows
                        ranges.append((size, size + part_size))
                        size += part_size
                    value["labels"] = dd.from_map(lambda r: pd.RangeIndex(*r), ranges)
            else:
                value.reset_index(drop=False, inplace=True)

    if value is None:
        value = type(df)()
        for c in selection_r.names:
            typ = df[selection_l.names[0]].dtype if len(selection_l.names) == 1 else df[c].dtype
            value[c] = nullable_series([None], df, typ)
        value.index = value.index.rename("labels")
        value.reset_index(drop=False, inplace=True)

    use_collection = isinstance(value, DaskDataFrame)
    if use_collection and value.npartitions == 1:
        # Use simple merge for single-partition case
        value = _compute_sync(value)
        use_collection = False

    # Determine encoding offsets
    null_encoding_offset = value["labels"].head(1).iloc[0] if single_table else NULL_OFFSET
    bucket_encoding_offset = null_encoding_offset + 1  # 2 (if not single_table)
    distinct_encoding_offset = bucket_encoding_offset + num_oov_buckets

    # Determine indices of "real" null values
    # (these will always be encoded to `1`)
    expr = df[selection_l.names[0]].isna()
    for _name in selection_l.names[1:]:
        expr = expr & df[_name].isna()
    nulls = df[expr].index

    if use_collection or not search_sorted:
        if list_col:
            codes = dispatch.flatten_list_column(df[selection_l.names[0]])
            codes["order"] = dispatch.arange(len(codes), like_df=df)
        else:
            # We go into this case
            codes = type(df)({"order": dispatch.arange(len(df), like_df=df)}, index=df.index)

        for cl, cr in zip(selection_l.names, selection_r.names):
            if isinstance(df[cl].dropna().iloc[0], (np.ndarray, list)):
                ser = df[cl].copy()
                codes[cl] = dispatch.flatten_list_column_values(ser).astype(value[cr].dtype)
            else:
                codes[cl] = df[cl].copy().astype(value[cr].dtype)

        indistinct = bucket_encoding_offset
        if buckets and storage_name in buckets:
            # apply hashing for "infrequent" categories
            indistinct = (
                _hash_bucket(df, buckets, selection_l.names, encode_type=encode_type)
                + bucket_encoding_offset
            )

            if use_collection:
                # Manual broadcast merge
                merged_df = _concat(
                    [
                        codes.merge(
                            _compute_sync(part),
                            left_on=selection_l.names,
                            right_on=selection_r.names,
                            how="left",
                        ).dropna(subset=["labels"])
                        for part in value.partitions
                    ],
                    ignore_index=False,
                ).sort_values("order")
            else:
                merged_df = codes.merge(
                    value, left_on=selection_l.names, right_on=selection_r.names, how="left"
                ).sort_values("order")

            merged_df.reset_index(drop=True, inplace=True)
            if len(merged_df) < len(codes):
                # Missing nulls
                labels = df._constructor_sliced(indistinct)
                labels.iloc[merged_df["order"]] = merged_df["labels"]
                labels = labels.values
            else:
                merged_df["labels"].fillna(df._constructor_sliced(indistinct), inplace=True)
                labels = merged_df["labels"].values
        else:
            # no hashing
            if use_collection:
                # Manual broadcast merge
                merged_df = _concat(
                    [
                        codes.merge(
                            _compute_sync(part),
                            left_on=selection_l.names,
                            right_on=selection_r.names,
                            how="left",
                        ).dropna(subset=["labels"])
                        for part in value.partitions
                    ],
                    ignore_index=True,
                )
                if len(merged_df) < len(codes):
                    # Missing nulls
                    labels = codes._constructor_sliced(
                        np.full(
                            len(codes),
                            indistinct,
                            like=merged_df["labels"].values,
                        ),
                    )
                    labels.iloc[merged_df["order"]] = merged_df["labels"]
                else:
                    labels = merged_df.sort_values("order")["labels"].reset_index(drop=True)
            else:
                labels = codes.merge(
                    value, left_on=selection_l.names, right_on=selection_r.names, how="left"
                ).sort_values("order")["labels"]
            labels.fillna(indistinct, inplace=True)
            labels = labels.values
    else:
        # Use `searchsorted` if we are using a "full" encoding
        if list_col:
            labels = (
                value[selection_r.names].searchsorted(
                    df[selection_l.names[0]].list.leaves, side="left", na_position="first"
                )
                + distinct_encoding_offset
            )
        else:
            labels = (
                value[selection_r.names].searchsorted(
                    df[selection_l.names], side="left", na_position="first"
                )
                + distinct_encoding_offset
            )
        labels[labels >= len(value[selection_r.names])] = bucket_encoding_offset

    # Make sure nulls are encoded to `null_encoding_offset`
    # (This should be `1` in most casese)
    if len(nulls):
        labels[nulls] = null_encoding_offset

    if list_col:
        labels = dispatch.encode_list_column(df[selection_l.names[0]], labels, dtype=dtype)
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


def is_list_col(col_selector, df):
    if isinstance(col_selector, list):
        col_selector = ColumnSelector(col_selector)
    has_lists = any(dispatch.is_list_dtype(df[col]) for col in col_selector.names)
    if has_lists and len(col_selector.names) != 1:
        raise ValueError("Can't categorical encode multiple list columns")
    return has_lists


def _maybe_flatten_list_column(col: str, df):
    # Flatten the specified column (col) if it is
    # a list dtype. Otherwise, pass back df "as is"
    selector = ColumnSelector([col])
    if is_list_col(selector, df):
        return dispatch.flatten_list_column(df[selector.names[0]])
    return df


def _hash_bucket(df, num_buckets, col, encode_type="joint"):
    if encode_type == "joint":
        nb = num_buckets[col[0]]
        encoded = dispatch.hash_series(df[col[0]]) % nb
    elif encode_type == "combo":
        if len(col) > 1:
            name = _make_name(*tuple(col), sep="_")
        else:
            name = col[0]
        nb = num_buckets[name]
        val = 0
        for column in col:
            val ^= dispatch.hash_series(df[column])  # or however we want to do this aggregation
        val = val % nb
        encoded = val
    return encoded


def _copy_storage(existing_stats, existing_path, new_path, copy):
    """helper function to copy files to a new storage location"""
    existing_fs = get_fs_token_paths(existing_path)[0]
    new_fs = get_fs_token_paths(new_path)[0]
    new_locations = {}
    for column, existing_file in existing_stats.items():
        new_file = existing_file.replace(str(existing_path), str(new_path))
        if copy and new_file != existing_file:
            new_fs.makedirs(os.path.dirname(new_file), exist_ok=True)
            with new_fs.open(new_file, "wb") as output:
                output.write(existing_fs.open(existing_file, "rb").read())

        new_locations[column] = new_file

    return new_locations


def _reset_df_index(col_name, cat_file_path, idx_count):
    cat_df = _compute_sync(dispatch.read_dispatch(collection=True)(cat_file_path))
    # change indexes for category
    cat_df.index = cat_df.index + idx_count
    # update count
    idx_count += cat_df.shape[0]
    # save the new indexes in file
    new_cat_file_path = _save_encodings(
        cat_df,
        Path(cat_file_path).parent,
        col_name,
        preserve_index=True,
    )
    return idx_count, new_cat_file_path


def _deprecate_tree_width(tree_width):
    # Warn user if tree_width is specified
    if tree_width is not None:
        warnings.warn(
            "The tree_width argument is now deprecated, and will be ignored. "
            "Please use split_out and split_every.",
            FutureWarning,
        )


def _compute_sync(collection):
    # Simple utility to compute a dask collection with
    # a synchronous scheduler (and to catch warnings
    # that are intended for users doing this by accident)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Running on a single-machine scheduler.*")
        return collection.compute(scheduler="synchronous")
