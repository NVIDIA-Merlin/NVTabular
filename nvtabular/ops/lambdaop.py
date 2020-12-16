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
from inspect import signature

import cudf
from nvtx import annotate

from .operator import Operator


class LambdaOp(Operator):
    """
    LambdaOp allows you to apply row level functions to an NVTabular workflow.

    Example usage 1::

        # Initialize the workflow
        proc = nvt.Workflow(
            cat_names=CATEGORICAL_COLUMNS,
            cont_names=CONTINUOUS_COLUMNS,
            label_name=LABEL_COLUMNS
        )

        # Add LamdaOp to the workflow and
        # specify an op_name
        # define a custom function e.g. extract first 5 character from string
        proc.add_feature(
            LambdaOp(
                op_name='first5char', #op_name - merged with column name
                f=lambda col: col.str.slice(0,5), # custom function
                columns=['cat1', 'cat2', 'cat3'], # columns, f is applied to
                replace=False # if columns will be replaced
            )
        )

    Example usage 2::

        # Add LambdaOp to the workflow and
        # specify an op_name
        # define a custom function e.g. calculate probability
        # for different events.
        proc.add_feature(
            LambdaOp(
                op_name='cond_prob', #op_name - merged with column name
                f=lambda col, gdf: col.astype(np.float32) / gdf['total_events'], # custom function
                columns=['event1', 'event2', 'event3'], # columns, f is applied to
                replace=False # if columns will be replaced
            )
        )

    Parameters
    -----------
    op_name : str:
        name of the operator column. It is used as a post_fix for the modified column names
        (if replace=False).
    f : callable
        Defines a function that takes a cudf.Series and an optional cudf.DataFrame as input,
        and returns a new Series as the output.
    columns :
        Columns to target for this op. If None, this operator will target all columns.
    replace : bool, default True
        Whether to replace existing columns or create new ones.
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
    def transform(self, columns, gdf: cudf.DataFrame):
        new_gdf = cudf.DataFrame()
        for col in columns:
            if self._param_count == 2:
                new_gdf[col] = self.f(gdf[col], gdf)
            elif self._param_count == 1:
                new_gdf[col] = self.f(gdf[col])
            else:
                # shouldn't ever happen,
                raise RuntimeError(f"unhandled lambda param count {self._param_count}")
        return new_gdf

    def dependencies(self):
        return self.dependency
