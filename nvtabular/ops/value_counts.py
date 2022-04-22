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

from merlin.core.dispatch import DataFrameType, is_list_dtype, pull_apart_list

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
            series = ddf[col]
            if is_list_dtype(series.compute()):
                stats[col] = stats[col] if col in stats else {}
                stats[col]["value_count"] = (
                    {} if "value_count" not in stats[col] else stats[col]["value_count"]
                )
                offs = pull_apart_list(series.compute())[1]
                lh, rh = offs[1:], offs[:-1]
                rh = rh.reset_index(drop=True)
                lh = lh.reset_index(drop=True)
                deltas = lh - rh
                # must be regular python class otherwise protobuf fails
                stats[col]["value_count"]["min"] = int(deltas.min())
                stats[col]["value_count"]["max"] = int(deltas.max())
        return stats

    def fit_finalize(self, dask_stats):
        self.stats = dask_stats

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        return df

    def _compute_properties(self, col_schema, input_schema):
        new_schema = super()._compute_properties(col_schema, input_schema)
        stat_properties = self.stats.get(col_schema.name, {})
        return col_schema.with_properties({**new_schema.properties, **stat_properties})

    def clear(self):
        self.stats = {}
