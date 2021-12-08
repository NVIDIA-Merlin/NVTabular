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
import os

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa
from nvtabular.graph.graph import Graph  # noqa
from nvtabular.graph.ops.selection import SelectionOp  # noqa
from nvtabular.inference.graph.ops.tensorflow import TensorflowOp  # noqa
from nvtabular.inference.graph.ops.workflow import WorkflowOp  # noqa


class Ensemble:
    def __init__(self, ops, schema, name="ensemble_model", label_columns=None):
        self.graph = Graph(ops)
        self.graph.fit_schema(schema)
        self.name = name
        self.label_columns = label_columns or []

    def export(self, export_path, version=1):
        parents = self.graph.output_node.parents_with_dependencies
        assert len(parents) == 1

        workflow_node = parents[0]
        assert len(workflow_node.parents_with_dependencies) == 1
        assert isinstance(workflow_node.op, WorkflowOp)

        selection_node = workflow_node.parents_with_dependencies[0]
        assert len(selection_node.parents_with_dependencies) == 0
        assert isinstance(selection_node.op, SelectionOp)

        model_node = self.graph.output_node
        assert isinstance(model_node.op, TensorflowOp)

        nodes_list = [model_node, workflow_node]
        config = None
        configs = []
        for node in nodes_list:
            config = node.op.export(export_path, config, version=version)
            configs.append(config)

        # generate the triton ensemble
        ensemble_path = os.path.join(export_path, self.name)
        os.makedirs(ensemble_path, exist_ok=True)
        os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)
        configs.reverse()
        return self._generate_ensemble_config(self.name, ensemble_path, configs)

    def _generate_ensemble_config(self, name, output_path, configs, name_ext=""):
        # TODO: max batchsize only relevant for workflow nodes
        ensemble_config = model_config.ModelConfig(
            name=name + name_ext, platform="ensemble", max_batch_size=configs[0].max_batch_size
        )
        ensemble_config.input.extend(configs[0].input)
        ensemble_config.output.extend(configs[-1].output)

        # workflow, model
        prev_step = None
        for idx, config in enumerate(configs):
            config_step = model_config.ModelEnsembling.Step(
                model_name=config.name, model_version=-1
            )
            for input_col in config.input:
                in_suffix = f"_{idx}" if idx > 0 else ""
                prev_step_ouputs = dict(prev_step.output_map) if prev_step else {}
                prev_step_input_col = (
                    prev_step_ouputs[input_col.name]
                    if prev_step_ouputs and input_col.name in prev_step_ouputs
                    else input_col.name
                )
                config_step.input_map[input_col.name] = prev_step_input_col + in_suffix
            for output_col in config.output:
                out_suffix = f"_{idx + 1}" if idx < len(configs) - 1 else ""
                config_step.output_map[output_col.name] = output_col.name + out_suffix
            ensemble_config.ensemble_scheduling.step.append(config_step)
            prev_step = config_step

        with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
            text_format.PrintMessage(ensemble_config, o)
        return ensemble_config
