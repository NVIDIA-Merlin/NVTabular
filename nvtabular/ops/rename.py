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
import cudf

from .operator import ColumnNames, Operator


class Rename(Operator):
    """This operation renames columns, either by using a user defined lambda function to
    transform column names, or by appending a postfix string to every column name

    Example usage::

        # Rename columns after LogOp
        cont_features = cont_names >> nvt.ops.LogOp() >> Rename(postfix='_log')
        processor = nvt.Workflow(cont_features)

    Parameters
    ----------
    f : callable, optional
        Function that takes a column name and returns a new column name
    postfix : str, optional
        If set each column name in the output will have this string appended to it
    """

    def __init__(self, f=None, postfix=None):
        if not f and postfix is None:
            raise ValueError("must specify either f or postfix for Rename op")

        self.postfix = postfix
        self.f = f

    def transform(self, columns: ColumnNames, gdf: cudf.DataFrame) -> cudf.DataFrame:
        gdf.columns = self.output_column_names(columns)
        return gdf

    transform.__doc__ = Operator.transform.__doc__

    def output_column_names(self, columns):
        if self.f:
            return [self.f(col) for col in columns]
        elif self.postfix:
            return [col + self.postfix for col in columns]
        else:
            raise RuntimeError("invalid rename op state found")
