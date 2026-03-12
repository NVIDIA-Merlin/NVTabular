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
from merlin.dag.ops.udf import UDF


class LambdaOp(UDF):
    """Apply a user-defined function to selected columns.

    This is a thin subclass of :class:`merlin.dag.ops.udf.UDF` that exists
    so NVTabular workflows have a stable import path for this operator.

    Serialization limitations
    -------------------------
    The JSON-based workflow serializer (used by ``Workflow.save`` /
    ``Workflow.load``) can only serialize **named functions** that are defined
    in an importable module.  Lambda functions and functions defined in
    ``__main__`` (e.g. in a script or Jupyter notebook) will raise a
    :class:`~nvtabular.workflow.graph_serializer.WorkflowSerializationError`
    at save time.

    To make a workflow with a ``LambdaOp`` serializable, replace any lambda
    with a named function in an importable module::

        # Not serializable:
        workflow = Workflow(["col"] >> LambdaOp(lambda x: x * 2))

        # Serializable (define in e.g. my_transforms.py):
        # my_transforms.py
        def double(x):
            return x * 2

        # your script / notebook:
        from my_transforms import double
        workflow = Workflow(["col"] >> LambdaOp(double))

    The function will be **re-imported by reference** at load time, so it must
    remain available and importable on the loading system.
    """
