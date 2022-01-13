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
import numpy as np

from nvtabular.dispatch import DataFrameType

from .operator import ColumnSelector, Operator


class ModelPredict(Operator):
    """
    Add a new column, corresponding to the result of calling
    ``model.predict`` on the Dataset for a specified model.

    Parameters
    -----------
    model : str or obj
        The model path, or explicit model object.
    column_name : str
        The name of the new prediction column.
    data_iterator_func : Callable; Optional
        Callable function to initialize the data-loading object
        used to feed each partition into ``model_predict_func``.
        Note that ``data_iterator_func`` must accept the input
        partition as the first positional argument, and must return
        an object that allows for iteration. Default behavior will
        result in a single iteration, with the entire partition
        being passed to ``model``.
    model_load_func : Callable; Optional
        Callable function to use for loading a model from disk.
        Default is a no-op.
    model_predict_func : Callable; Optional
        Function wrapper to use for each batch prediction. This
        specified function must have the signature ``(model, batch)``.
        Default will return ``model(batch)``.
    predict_concat_func : Callable; Optional
        Callable function to apply to the list of predictions for
        each batch. This function must concatenate the list of results
        into a single series-like object for column assignment.
        Default is ``numpy.concatenate``.
    """

    def __init__(
        self,
        model,
        column_name,
        data_iterator_func=None,
        model_load_func=None,
        model_predict_func=None,
        predict_concat_func=None,
    ):
        super().__init__()
        self._model = model
        self.column_name = column_name
        self.data_iterator_func = data_iterator_func
        self.model_predict_func = model_predict_func
        self.model_load_func = model_load_func
        self.predict_concat_func = predict_concat_func

    @property
    def model(self):
        if isinstance(self._model, str):
            self._model = self.model_load_func(self._model)
        return self._model

    def transform(self, col_selector: ColumnSelector, df: DataFrameType) -> DataFrameType:

        # Set defaults
        iterator_func = self.data_iterator_func or (lambda x: [x])
        predict_func = self.model_predict_func or (lambda x, y: x(y))
        concat_func = self.predict_concat_func or np.concatenate

        # Iterate over batches of df and collect predictions
        new_df = df[col_selector.names]
        new_df[self.column_name] = concat_func(
            [predict_func(self.model, batch) for batch in iterator_func(df)]
        )

        # Return result
        return new_df

    def column_mapping(self, col_selector):
        column_mapping = super().column_mapping(col_selector)
        column_mapping[self.column_name] = col_selector.names
        return column_mapping

    transform.__doc__ = Operator.transform.__doc__
