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

from nvtabular.dispatch import DataFrameType, _natural_log, annotate

from .operator import ColumnNames, Operator


class LogOp(Operator):
    """
    This operator calculates the log of continuous columns. Note that
    to handle the common case of zerofilling null values, this
    calculates ``log(1+x)`` instead of just ``log(x)``.

    Example usage::

        # Use LogOp to define NVTabular workflow
        cont_features = cont_names >> nvt.ops.LogOp() >> ...
        processor = nvt.Workflow(cont_features)
    """

    @annotate("LogOp_op", color="darkgreen", domain="nvt_python")
    def transform(self, columns: ColumnNames, df: DataFrameType) -> DataFrameType:
        return _natural_log(df[columns].astype(np.float32) + 1)

    transform.__doc__ = Operator.transform.__doc__
