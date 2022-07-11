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
import re

import numpy
from dask.dataframe.utils import meta_nonempty

from merlin.core.dispatch import DataFrameType, annotate
from merlin.schema import Schema

from .operator import ColumnSelector, Operator


class Groupby(Operator):
    """Groupby Transformation

    Locally transform each partition of a Dataset with one or
    more groupby aggregations.

    WARNING: This transformation does NOT move data between
    partitions. Please make sure that the target Dataset object
    is already shuffled by ``groupby_cols``, otherwise the
    output may be incorrect. See: ``Dataset.shuffle_by_keys``.

    Example usage::

        groupby_cols = ['user_id', 'session_id']
        dataset = dataset.shuffle_by_keys(keys=groupby_cols)

        groupby_features = [
            'user_id', 'session_id', 'month', 'prod_id',
        ] >> ops.Groupby(
            groupby_cols=groupby_cols,
            sort_cols=['month'],
            aggs={
                'prod_id': 'list',
                'month': ['first', 'last'],
            },
        )
        processor = nvtabular.Workflow(groupby_features)

        workflow.fit(dataset)
        dataset_transformed = workflow.transform(dataset)

    Parameters
    -----------
    groupby_cols : str or list of str
        The column names to be used as groupby keys.
        WARNING: Ensure the dataset was partitioned by those
        groupby keys (see above for an example).
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

    def __init__(
        self, groupby_cols=None, sort_cols=None, aggs="list", name_sep="_", ascending=True
    ):
        self.groupby_cols = groupby_cols
        self.sort_cols = sort_cols or []
        if isinstance(self.groupby_cols, str):
            self.groupby_cols = [self.groupby_cols]
        if isinstance(self.sort_cols, str):
            self.sort_cols = [self.sort_cols]
        self.ascending = ascending

        # Split aggregations into "conventional" aggregations
        # and "list-based" aggregations.  After this block,
        # we will have a dictionary for each of these cases.
        # We use the "__all__" key to specify aggregations
        # that will be performed on all (non-key) columns.
        self.list_aggs, self.conv_aggs = {}, {}
        if isinstance(aggs, str):
            aggs = {"__all__": [aggs]}
        elif isinstance(aggs, list):
            aggs = {"__all__": aggs}
        for col, v in aggs.items():
            _aggs = v if isinstance(v, list) else [v]
            _conv_aggs, _list_aggs = set(), set()
            for _agg in _aggs:
                if is_list_agg(_agg):
                    _list_aggs.add("list" if _agg == list else _agg)
                    _conv_aggs.add(list)
                else:
                    _conv_aggs.add(_agg)
            if _conv_aggs:
                self.conv_aggs[col] = list(_conv_aggs)
            if _list_aggs:
                self.list_aggs[col] = list(_list_aggs)

        self.name_sep = name_sep
        super().__init__()

    @annotate("Groupby_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        # Sort if necessary
        if self.sort_cols:
            df = df.sort_values(self.sort_cols, ascending=self.ascending, ignore_index=True)

        # List aggregations do not work with empty data.
        # Use synthetic metadata to predict output columns.
        empty_df = not len(df)

        _df = meta_nonempty(df) if empty_df else df

        # Get "complete" aggregation dicts
        _list_aggs, _conv_aggs = _get_agg_dicts(
            self.groupby_cols, self.list_aggs, self.conv_aggs, col_selector
        )

        # Apply aggregations
        new_df = _apply_aggs(
            _df,
            self.groupby_cols,
            _list_aggs,
            _conv_aggs,
            name_sep=self.name_sep,
            ascending=self.ascending,
        )

        if empty_df:
            return new_df.iloc[:0]
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    def compute_output_schema(
        self, input_schema: Schema, col_selector: ColumnSelector, prev_output_schema: Schema = None
    ) -> Schema:
        if not col_selector and hasattr(self, "target"):
            col_selector = (
                ColumnSelector(self.target) if isinstance(self.target, list) else self.target
            )
        return super().compute_output_schema(input_schema, col_selector, prev_output_schema)

    def column_mapping(self, col_selector):
        column_mapping = {}

        for groupby_col in self.groupby_cols:
            if groupby_col in col_selector.names:
                column_mapping[groupby_col] = [groupby_col]

        _list_aggs, _conv_aggs = _get_agg_dicts(
            self.groupby_cols, self.list_aggs, self.conv_aggs, col_selector
        )

        for input_col_name, aggs in _list_aggs.items():
            output_col_names = _columns_out_from_aggs(
                {input_col_name: aggs}, name_sep=self.name_sep
            )
            for output_col_name in output_col_names:
                column_mapping[output_col_name] = [input_col_name]

        for input_col_name, aggs in _conv_aggs.items():
            output_col_names = _columns_out_from_aggs(
                {input_col_name: aggs}, name_sep=self.name_sep
            )
            for output_col_name in output_col_names:
                column_mapping[output_col_name] = [input_col_name]

        return column_mapping

    @property
    def dependencies(self):
        return self.groupby_cols

    def _compute_dtype(self, col_schema, input_schema):
        col_schema = super()._compute_dtype(col_schema, input_schema)

        dtype = col_schema.dtype
        is_list = col_schema.is_list

        dtypes = {
            "count": numpy.int32,
            "nunique": numpy.int32,
            "mean": numpy.float32,
            "var": numpy.float32,
            "std": numpy.float32,
            "median": numpy.float32,
        }

        is_lists = {"list": True}

        for col_name in input_schema.column_names:
            combined_aggs = _aggs_for_column(col_name, self.conv_aggs)
            combined_aggs += _aggs_for_column(col_name, self.list_aggs)
            for agg in combined_aggs:
                if col_schema.name.endswith(f"{self.name_sep}{agg}"):
                    dtype = dtypes.get(agg, dtype)
                    is_list = is_lists.get(agg, is_list)
                    break

        return col_schema.with_dtype(dtype, is_list=is_list, is_ragged=is_list)


def _aggs_for_column(col_name, agg_dict):
    return agg_dict.get(col_name, []) + agg_dict.get("__all__", [])


def _columns_out_from_aggs(aggs, name_sep="_"):
    # Helper function for `output_column_names`
    _agg_cols = []
    for k, v in aggs.items():
        for _v in v:
            if isinstance(_v, str):
                _agg_cols.append(name_sep.join([k, _v]))
    return _agg_cols


def _apply_aggs(_df, groupby_cols, _list_aggs, _conv_aggs, name_sep="_", ascending=True):

    # Apply conventional aggs
    _columns = list(set(groupby_cols) | set(_conv_aggs) | set(_list_aggs))
    df = _df[_columns].groupby(groupby_cols).agg(_conv_aggs).reset_index()

    df.columns = [
        name_sep.join([n for n in name if n != ""]) for name in df.columns.to_flat_index()
    ]

    # Handle custom aggs (e.g. "first" and "last")
    for col, aggs in _list_aggs.items():
        for _agg in aggs:
            if is_list_agg(_agg, custom=True):
                df[f"{col}{name_sep}{_agg}"] = _first_or_last(
                    df[f"{col}{name_sep}list"], _agg, ascending=ascending
                )
        if "list" not in aggs:
            df.drop(columns=[col + f"{name_sep}list"], inplace=True)

    for col in df.columns:
        if re.search(f"{name_sep}(count|nunique)$", col):
            df[col] = df[col].astype(numpy.int32)
        elif re.search(f"{name_sep}(mean|median|std|var)$", col):
            df[col] = df[col].astype(numpy.float32)

    return df


def _get_agg_dicts(groupby_cols, list_aggs, conv_aggs, columns):
    # Get updated aggregation dicts. This should map "__all__"
    # to specific columns, and remove elements that are not
    # in `columns`.
    _allowed_cols = [c for c in columns.names if c not in groupby_cols]
    _list_aggs = _ensure_agg_dict(list_aggs, _allowed_cols)
    _conv_aggs = _ensure_agg_dict(conv_aggs, _allowed_cols)
    return _list_aggs, _conv_aggs


def _ensure_agg_dict(_aggs, _allowed_cols):
    # Make sure aggregation dict has legal keys
    if "__all__" in _aggs:
        return {col: _aggs["__all__"] for col in _allowed_cols}
    else:
        return {k: v for k, v in _aggs.items() if k in _allowed_cols}


def is_list_agg(agg, custom=False):
    # check if `agg` is a supported list aggregation
    if custom:
        return agg in ("first", "last")
    else:
        return agg in ("list", list, "first", "last")


def _first_or_last(x, kind, ascending=True):
    # Redirect to _first or _last
    if kind == "first" and ascending:
        return _first(x)
    elif kind == "last" and not ascending:
        return _first(x)
    else:
        return _last(x)


def _first(x):
    # Convert each element of a list column to be the first
    # item in the list
    if hasattr(x, "list"):
        # cuDF-specific behavior
        offsets = x.list._column.offsets
        elements = x.list._column.elements
        return elements[offsets[:-1]]
    else:
        # cpu/pandas
        return x.apply(lambda y: y[0])


def _last(x):
    # Convert each element of a list column to be the last
    # item in the list
    if hasattr(x, "list"):
        # cuDF-specific behavior
        offsets = x.list._column.offsets
        elements = x.list._column.elements
        return elements[offsets[1:].values - 1]
    else:
        # cpu/pandas
        return x.apply(lambda y: y[-1])
