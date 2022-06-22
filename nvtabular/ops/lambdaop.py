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
from inspect import getsourcelines, signature

from merlin.core.dispatch import DataFrameType, annotate

from .operator import ColumnSelector, Operator


class LambdaOp(Operator):
    """
    LambdaOp allows you to apply row level functions to an NVTabular workflow.

    Example usage 1::

        # Define a ColumnSelector that LamdaOp will apply to
        # then define a custom function, e.g. extract first 5 character from a string
        lambda_feature = ColumnSelector(["col1"])
        new_lambda_feature = lambda_feature >> LambdaOp(lambda col: col.str.slice(0, 5))
        workflow = nvtabular.Workflow(new_lambda_feature + 'label')

    Example usage 2::

        # define a custom function e.g. calculate probability for different events.
        # Rename the each new feature column name.
        lambda_features = ColumnSelector(['event1', 'event2', 'event3']), # columns, f is applied to
        def cond_prob(col, gdf):
            col = col.astype(np.float32)
            col = col / gdf['total_events']
            return col
        new_lambda_features = lambda_features >> LambdaOp(cond_prob, dependency=["total_events"]) \
>> Rename(postfix="_cond")
        workflow = nvtabular.Workflow(new_lambda_features + 'label')

    Parameters
    -----------
    f : callable
        Defines a function that takes a Series and an optional DataFrame as input,
        and returns a new Series as the output.
    dependency : list, default None
        Whether to provide a dependency column or not.
    """

    def __init__(self, f, dependency=None, label=None, dtype=None, tags=None, properties=None):
        super().__init__()
        if f is None:
            raise ValueError("f cannot be None. LambdaOp op applies f to dataframe")
        self.f = f
        self._param_count = len(signature(self.f).parameters)
        if self._param_count not in (1, 2):
            raise ValueError("lambda function must accept either one or two parameters")
        self.dependency = dependency
        self._label = label

        self._dtype = dtype
        self._tags = tags or []
        self._properties = properties or {}

    @annotate("DFLambda_op", color="darkgreen", domain="nvt_python")
    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:
        new_df = type(df)()
        for col in col_selector.names:
            if self._param_count == 2:
                new_df[col] = self.f(df[col], df)
            elif self._param_count == 1:
                new_df[col] = self.f(df[col])
            else:
                # shouldn't ever happen,
                raise RuntimeError(f"unhandled lambda param count {self._param_count}")
        return new_df

    transform.__doc__ = Operator.transform.__doc__

    @property
    def dependencies(self):
        return self.dependency

    @property
    def label(self):
        # if we're given an explicit label to use, return it
        if self._label:
            return self._label

        # if we have a named function (not a lambda) return the function name
        name = self.f.__name__
        if name != "<lambda>":
            return name
        else:
            # otherwise get the lambda source code from the inspect module if possible
            source = getsourcelines(self.f)[0][0]
            lambdas = [op.strip() for op in source.split(">>") if "lambda " in op]
            if len(lambdas) == 1 and lambdas[0].count("lambda") == 1:
                return lambdas[0]

        # Failed to figure out the source
        return "LambdaOp"

    def column_mapping(self, col_selector):
        filtered_selector = self._remove_deps(col_selector, self.dependencies)
        return super().column_mapping(filtered_selector)

    def _remove_deps(self, col_selector, dependencies):
        dependencies = dependencies or []
        to_skip = ColumnSelector(
            [
                dep if isinstance(dep, str) else dep.output_schema.column_names
                for dep in dependencies
            ]
        )
        return col_selector.filter_columns(to_skip)

    @property
    def dynamic_dtypes(self):
        return True

    @property
    def output_dtype(self):
        return self._dtype

    @property
    def output_tags(self):
        return self._tags

    @property
    def output_properties(self):
        return self._properties
