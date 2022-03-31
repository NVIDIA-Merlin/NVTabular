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
from merlin.dag import Node
from nvtabular.ops import LambdaOp, Operator


class WorkflowNode(Node):
    """WorkflowNode represents a Node in a NVTabular workflow graph"""

    def __rshift__(self, operator):
        if callable(operator) and not (
            isinstance(operator, type) and issubclass(operator, Operator)
        ):
            # implicit lambdaop conversion.
            operator = LambdaOp(operator)

        return super().__rshift__(operator)
