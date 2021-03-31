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
from collections import defaultdict

import cudf
from dask.dataframe.utils import meta_nonempty
from nvtx import annotate

from .operator import ColumnNames, Operator


class Groupby(Operator):
    """Local Groupby Transformation

    WARNING: This transformation does NOT transfer data between
    partitions. Please make sure that the target Dataset object is
    already shuffled by the desired groupby keys, otherwise the
    output of this transformation may be incorrect.

    See: ``Dataset.shuffle_by_keys``.

    Example usage::

        groupby_features = [
            'user_id', 'session_id', 'month', 'prod_id',
        ] >> ops.Groupby(
            groupby_cols=['user_id', 'session_id'],
            sort_cols=['month'],
            aggs={
                'prod_id': 'list',
                'month': ['first', 'last'],
            },
        )
        processor = nvtabular.Workflow(groupby_features)

    Parameters
    -----------
    groupby_cols : str or list of str
        The column names to be used as groupby keys.
    sort_cols : str or list of str
        Columns to be used to sort each partition before
        groupby aggregation is performed. If this argument
        is not specified, the results will not be sorted.
    aggs : dict, list or str
        Groupby aggregations to perform. Supported list-based
        aggregations include "list", "first" & "last". Most
        conventional aggregations supported by Pandas/cuDF are
        also allowed (e.g. "sum", "count", "max", "mean", etc.).
    name_sep : str
        String separator to use for new column names.
    """

    def __init__(self, groupby_cols=None, sort_cols=None, aggs="list", name_sep="_"):
        self.groupby_cols = groupby_cols
        self.sort_cols = sort_cols or []
        if isinstance(self.groupby_cols, str):
            self.groupby_cols = [self.groupby_cols]
        if isinstance(self.sort_cols, str):
            self.sort_cols = [self.sort_cols]

        # Split aggregations into "conventional" aggregations
        # and "list-based" aggregations.  After this block,
        # we will have a dictionary for each of these cases.
        # We use the "__all__" key to specify aggregations
        # that will be performed on all (non-key) columns.
        self.list_aggs = defaultdict(list)
        self.conv_aggs = defaultdict(list)
        if isinstance(aggs, str):
            aggs = {"__all__": [aggs]}
        elif isinstance(aggs, list):
            aggs = {"__all__": aggs}
        for col, v in aggs.items():
            _aggs = v if isinstance(v, list) else [v]
            for _agg in _aggs:
                if _is_list_agg(_agg):
                    self.list_aggs[col].append(_agg)
                else:
                    self.conv_aggs[col].append(_agg)

        self.name_sep = name_sep
        super().__init__()

    @annotate("Groupby_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, gdf: cudf.DataFrame) -> cudf.DataFrame:

        # Sort if necessary
        if self.sort_cols:
            gdf = gdf.sort_values(self.sort_cols, ignore_index=True)

        # List aggregations do not work with empty data.
        # Use synthetic metadata to predict output columns.
        empty_df = not len(gdf)
        _df = meta_nonempty(gdf) if empty_df else gdf

        # Get "complete" aggregation dicts
        _list_aggs, _conv_aggs = _get_agg_dicts(
            self.groupby_cols, self.list_aggs, self.conv_aggs, columns
        )

        # Get list-aggregation result
        df_la = _apply_list_aggs(_df, self.groupby_cols, _list_aggs, name_sep=self.name_sep)

        # Get conventional-aggregation result
        df_cv = _apply_conventional_aggs(_df, self.groupby_cols, _conv_aggs, name_sep=self.name_sep)

        if df_cv is None:
            # Only using list aggregations
            new_gdf = df_la
        elif df_la is None:
            # Only using conventional aggregations
            new_gdf = df_cv
        else:
            # Using both aggregations, merge results
            new_gdf = df_cv.merge(df_la, self.groupby_cols, how="outer")

        if empty_df:
            return new_gdf.iloc[:0]
        return new_gdf

    transform.__doc__ = Operator.transform.__doc__

    def output_column_names(self, columns):
        # Get the expected column names after transformation
        _list_aggs, _conv_aggs = _get_agg_dicts(
            self.groupby_cols, self.list_aggs, self.conv_aggs, columns
        )
        _list_aggs = _columns_out_from_aggs(_list_aggs, name_sep=self.name_sep)
        _conv_aggs = _columns_out_from_aggs(_conv_aggs, name_sep=self.name_sep)

        return list(set(self.groupby_cols) | set(_list_aggs) | set(_conv_aggs))


def _columns_out_from_aggs(aggs, name_sep="_"):
    # Helper function for `output_column_names`
    _agg_cols = []
    for k, v in aggs.items():
        for _v in v:
            _agg_cols.append(name_sep.join([k, _v]))
    return _agg_cols


def _apply_conventional_aggs(_df, groupby_cols, _conv_aggs, name_sep="_"):
    df_cv = None  # Default return
    if _conv_aggs:
        _columns = list(set(groupby_cols) | set(_conv_aggs))
        df_cv = _df[_columns].groupby(groupby_cols).agg(_conv_aggs).reset_index()
        df_cv.columns = [
            name_sep.join([n for n in name if n != ""]) for name in df_cv.columns.to_flat_index()
        ]
    return df_cv


def _apply_list_aggs(_df, groupby_cols, _list_aggs, name_sep="_"):
    df_la = None  # Default return
    if _list_aggs:

        # Perform initial "collect" aggregation
        _columns = list(set(groupby_cols) | set(_list_aggs))
        df_la = _df[_columns].groupby(groupby_cols).collect().reset_index()
        columns = [c + f"{name_sep}list" if c in _list_aggs else c for c in df_la.columns]
        df_la.columns = columns

        # Handle "first" and "last" aggregations
        for col, aggs in _list_aggs.items():
            for _agg in aggs:
                if _agg in ("first", "last"):
                    df_la[f"{col}{name_sep}{_agg}"] = _first_or_last(
                        df_la[f"{col}{name_sep}list"], _agg
                    )
            if "list" not in aggs:
                df_la.drop(columns=[col + f"{name_sep}list"], inplace=True)
    return df_la


def _get_agg_dicts(groupby_cols, list_aggs, conv_aggs, columns):
    # Get updated aggregation dicts. This should map "__all__"
    # to specific columns, and remove elements that are not
    # in `columns`.
    _allowed_cols = [c for c in columns if c not in groupby_cols]
    _list_aggs = _ensure_agg_dict(list_aggs, _allowed_cols)
    _conv_aggs = _ensure_agg_dict(conv_aggs, _allowed_cols)
    return _list_aggs, _conv_aggs


def _ensure_agg_dict(_aggs, _allowed_cols):
    # Make sure aggregation dict has legal keys
    if "__all__" in _aggs:
        return {col: _aggs["__all__"] for col in _allowed_cols}
    else:
        return {k: v for k, v in _aggs.items() if k in _allowed_cols}


def _is_list_agg(agg):
    # check if `agg` is a supported list aggregation
    return agg in ("list", "first", "last")


def _first_or_last(x, kind):
    # Redirect to _first or _last
    return _first(x) if kind == "first" else _last(x)


def _first(x):
    # Convert each element of a list column to be the first
    # item in the list
    offsets = x.list._column.offsets
    elements = x.list._column.elements
    return [elements[offsets[i]] for i in range(0, len(offsets) - 1)]


def _last(x):
    # Convert each element of a list column to be the last
    # item in the list
    offsets = x.list._column.offsets
    elements = x.list._column.elements
    return [elements[offsets[i] - 1] for i in range(1, len(offsets))]
