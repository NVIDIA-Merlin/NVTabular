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
from inspect import signature

from nvtx import annotate

from ..dispatch import DataFrameType
from .operator import ColumnNames, Operator


class LambdaOp(Operator):
    """
    LambdaOp allows you to apply row level functions to an NVTabular workflow.

    Example usage 1::

        # Define a ColumnGroup that LamdaOp will apply to
        # then define a custom function, e.g. extract first 5 character from a string
        lambda_feature = ColumnGroup(["col1"])
        new_lambda_feature = lambda_feature >> (lambda col: col.str.slice(0, 5))
        processor = nvtabular.Workflow(new_lambda_feature + 'label')

    Example usage 2::

        # define a custom function e.g. calculate probability for different events.
        # Rename the each new feature column name.
        lambda_features = ColumnGroup(['event1', 'event2', 'event3']), # columns, f is applied to
        def cond_prob(col, gdf):
            col = col.astype(np.float32)
            col = col / gdf['total_events']
            return col
        new_lambda_features = lambda_features >> LambdaOp(cond_prob, dependency=["total_events"]) \
>> Rename(postfix="_cond")
        processor = nvtabular.Workflow(new_lambda_features + 'label')

    Parameters
    -----------
    f : callable
        Defines a function that takes a cudf.Series and an optional cudf.DataFrame as input,
        and returns a new Series as the output.
    dependency : list, default None
        Whether to provide a dependency column or not.
    """

    def __init__(self, f, dependency=None):
        super().__init__()
        if f is None:
            raise ValueError("f cannot be None. LambdaOp op applies f to dataframe")
        self.f = f
        self._param_count = len(signature(self.f).parameters)
        if self._param_count not in (1, 2):
            raise ValueError("lambda function must accept either one or two parameters")
        self.dependency = dependency

    @annotate("DFLambda_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        new_df = type(df)()
        for col in columns:
            if self._param_count == 2:
                new_df[col] = self.f(df[col], df)
            elif self._param_count == 1:
                new_df[col] = self.f(df[col])
            else:
                # shouldn't ever happen,
                raise RuntimeError(f"unhandled lambda param count {self._param_count}")
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    def dependencies(self):
        return self.dependency
