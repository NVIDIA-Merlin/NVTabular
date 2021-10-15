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

from .operator import ColumnSelector, Operator


class StatOperator(Operator):
    """
    Base class for statistical operator classes. This adds a 'fit' and 'finalize' method
    on top of the Operator class.
    """

    def fit(self, col_selector: ColumnSelector, ddf: dd.DataFrame) -> Any:
        """Calculate statistics for this operator, and return a dask future
        to these statistics, which will be computed by the workflow."""

        raise NotImplementedError(
            """The dask operations needed to return a dictionary of uncomputed statistics."""
        )

    def fit_finalize(self, dask_stats):
        """Finalize statistics calculation - the workflow calls this function with
        the computed statistics from the 'fit' object'"""

        raise NotImplementedError(
            """Follow-up operations to convert dask statistics in to member variables"""
        )

    def clear(self):
        """zero and reinitialize all relevant statistical properties"""
        raise NotImplementedError("clear isn't implemented for this op!")

    def set_storage_path(self, new_path, copy=False):
        """Certain stat operators need external storage - for instance Categorify writes out
        parquet files containing the categorical mapping. When we save the operator, we
        also want to save these files as part of the bundle. Implementing this method
        lets statoperators bundle their dependent files into the new path that we're writing
        out (note that this could happen after the operator is created)
        """
