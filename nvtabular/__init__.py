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
import warnings

from . import graph, io, workflow  # noqa
from ._version import get_versions

# suppress some warnings with cudf warning about column ordering with dlpack
# and numba warning about deprecated environment variables
warnings.filterwarnings("ignore", module="cudf.io.dlpack")
warnings.filterwarnings("ignore", module="numba.cuda.envvars")


WorkflowNode = workflow.WorkflowNode
ColumnSelector = graph.selector.ColumnSelector
Workflow = workflow.Workflow
Dataset = io.dataset.Dataset
ColumnSchema = graph.schema.ColumnSchema
Schema = graph.schema.Schema


# Provides an alias of ColumnSelector so that old usages of ColumnGroup to
# select columns at the beginning of an operator chain don't break
def ColumnGroup(columns):
    warnings.warn("ColumnGroup is deprecated, use ColumnSelector instead", DeprecationWarning)
    return ColumnSelector(columns)


__all__ = [
    "Workflow",
    "Dataset",
    "WorkflowNode",
    "ColumnGroup",
    "ColumnSelector",
    "ColumnSchema",
    "Schema",
]

# cudf warns about column ordering with dlpack methods, ignore it
warnings.filterwarnings("ignore", module="cudf.io.dlpack")


__version__ = get_versions()["version"]
del get_versions
