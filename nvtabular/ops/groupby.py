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

from dask.dataframe.utils import meta_nonempty

from nvtabular.dispatch import DataFrameType, annotate

from .operator import ColumnNames, Operator


class Groupby(Operator):
    """Groupby Transformation

    Locally transform each partition of a Dataset with one or
    more groupby aggregations.

    WARNING: This transformation does NOT move data between
    partitions. Please make sure that the target Dataset object
    is already shuffled by ``groupby_cols``, otherwise the
    output may be incorrect. See: ``Dataset.shuffle_by_keys``.

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
        self.list_aggs, self.conv_aggs = {}, {}
        if isinstance(aggs, str):
            aggs = {"__all__": [aggs]}
        elif isinstance(aggs, list):
            aggs = {"__all__": aggs}
        for col, v in aggs.items():
            _aggs = v if isinstance(v, list) else [v]
            _conv_aggs, _list_aggs = set(), set()
            for _agg in _aggs:
                if _is_list_agg(_agg):
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
    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:

        # Sort if necessary
        if self.sort_cols:
            df = df.sort_values(self.sort_cols, ignore_index=True)

        # List aggregations do not work with empty data.
        # Use synthetic metadata to predict output columns.
        empty_df = not len(df)
        _df = meta_nonempty(df) if empty_df else df

        # Get "complete" aggregation dicts
        _list_aggs, _conv_aggs = _get_agg_dicts(
            self.groupby_cols, self.list_aggs, self.conv_aggs, columns
        )

        # Apply aggregations
        new_df = _apply_aggs(_df, self.groupby_cols, _list_aggs, _conv_aggs, name_sep=self.name_sep)

        if empty_df:
            return new_df.iloc[:0]
        return new_df

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
            if isinstance(_v, str):
                _agg_cols.append(name_sep.join([k, _v]))
    return _agg_cols


def _apply_aggs(_df, groupby_cols, _list_aggs, _conv_aggs, name_sep="_"):

    # Apply conventional aggs
    _columns = list(set(groupby_cols) | set(_conv_aggs) | set(_list_aggs))
    df = _df[_columns].groupby(groupby_cols).agg(_conv_aggs).reset_index()
    df.columns = [
        name_sep.join([n for n in name if n != ""]) for name in df.columns.to_flat_index()
    ]

    # Handle custom aggs (e.g. "first" and "last")
    for col, aggs in _list_aggs.items():
        for _agg in aggs:
            if _is_list_agg(_agg, custom=True):
                df[f"{col}{name_sep}{_agg}"] = _first_or_last(df[f"{col}{name_sep}list"], _agg)
        if "list" not in aggs:
            df.drop(columns=[col + f"{name_sep}list"], inplace=True)

    return df


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


def _is_list_agg(agg, custom=False):
    # check if `agg` is a supported list aggregation
    if custom:
        return agg in ("first", "last")
    else:
        return agg in ("list", list, "first", "last")


def _first_or_last(x, kind):
    # Redirect to _first or _last
    return _first(x) if kind == "first" else _last(x)


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
