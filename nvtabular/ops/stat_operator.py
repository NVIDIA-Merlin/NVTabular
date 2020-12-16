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
from .operator import Operator


class StatOperator(Operator):
    """
    Base class for statistical operator classes. This adds a 'fit' and 'finalize' method
    on top of
    """

    def __init__(self):
        super(StatOperator, self).__init__()

    def fit(self, columns, ddf):
        raise NotImplementedError(
            """The dask operations needed to return a dictionary of uncomputed statistics."""
        )

    def fit_finalize(self, dask_stats):
        raise NotImplementedError(
            """Follow-up operations to convert dask statistics in to member variables"""
        )

    def clear(self):
        raise NotImplementedError("""zero and reinitialize all relevant statistical properties""")
