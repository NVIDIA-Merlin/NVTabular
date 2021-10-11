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
#

from typing import Any

import dask.dataframe as dd

from nvtabular.dispatch import DataFrameType, _is_list_dtype, _pull_apart_list

from .operator import ColumnSelector
from .stat_operator import StatOperator


class ValueCount(StatOperator):
    """
    The operator calculates the min and max lengths of multihot columns.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stats = {}

    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame) -> Any:
        stats = {}
        for col in col_selector.names:
            series = ddf[col].compute()
            if _is_list_dtype(series):
                offs = series.list.len() if hasattr(series, "list") else _pull_apart_list(series)[1]
                stats[col] = stats[col] if col in stats else {}
                stats[col]["value_count"] = (
                    {} if "value_count" not in stats[col] else stats[col]["value_count"]
                )
                stats[col]["value_count"]["min"] = offs.min()
                stats[col]["value_count"]["max"] = offs.max()
        return stats

    def fit_finalize(self, dask_stats):
        self.stats = dask_stats

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df

    def output_properties(self):
        return self.stats
