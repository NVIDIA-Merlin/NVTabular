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

from nvtabular.dispatch import DataFrameType, annotate

from .operator import ColumnNames, Operator


class Dropna(Operator):
    """
    This operation detects and filters out rows with missing values.

    Example usage::

        # Use Dropna to define a NVTabular workflow
        # Default is None and will check all columns
        dropna_features = ['cat1', 'num1'] >> ops.Dropna() >> ...
        processor = nvtabular.Workflow(dropna_features)

    Parameters
    ----------
    """

    @annotate("Dropna_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        new_df = df.dropna(subset=columns or None)
        new_df.reset_index(drop=True, inplace=True)
        return new_df
