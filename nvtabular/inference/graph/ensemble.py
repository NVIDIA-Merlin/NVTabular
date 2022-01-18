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

from nvtabular.graph.node import postorder_iter_nodes

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa
from nvtabular.inference.graph.graph import InferenceGraph  # noqa
from nvtabular.inference.triton.ensemble import _convert_dtype  # noqa


class Ensemble:
    def __init__(self, ops, schema, name="ensemble_model", label_columns=None):
        self.graph = InferenceGraph(ops)
        self.graph.fit_schema(schema)
        self.name = name
        self.label_columns = label_columns or []

    def export(self, export_path, version=1):
        # Create ensemble config
        ensemble_config = model_config.ModelConfig(
            name=self.name,
            platform="ensemble",
            # max_batch_size=configs[0].max_batch_size
        )

        for col_name, col_schema in self.graph.input_schema.column_schemas.items():
            ensemble_config.input.append(
                model_config.ModelInput(
                    name=col_name, data_type=_convert_dtype(col_schema.dtype), dims=[-1, 1]
                )
            )

        for col_name, col_schema in self.graph.output_schema.column_schemas.items():
            ensemble_config.output.append(
                model_config.ModelOutput(
                    name=col_name, data_type=_convert_dtype(col_schema.dtype), dims=[-1, 1]
                )
            )

        # Build node id lookup table
        postorder_nodes = list(postorder_iter_nodes(self.graph.output_node))

        node_idx = 0
        node_id_lookup = {}
        for node in postorder_nodes:
            if hasattr(node.op, "export"):
                node_id_lookup[node] = node_idx
                node_idx += 1

        node_configs = []
        # Export node configs and add ensemble steps
        for node in postorder_nodes:
            if hasattr(node.op, "export"):
                node_config = node.op.export(export_path, version=version)

                config_step = model_config.ModelEnsembling.Step(
                    model_name=node.op.export_name, model_version=-1
                )

                for input_col_name in node.input_columns.names:
                    source = _find_column_source(node, input_col_name)
                    source_id = node_id_lookup.get(source, None)
                    in_suffix = f"_{source_id}" if source_id is not None else ""
                    config_step.input_map[input_col_name] = input_col_name + in_suffix

                for output_col_name in node.output_columns.names:
                    node_id = node_id_lookup.get(node, None)
                    out_suffix = (
                        f"_{node_id}" if node_id is not None and node_id < node_idx - 1 else ""
                    )
                    config_step.output_map[output_col_name] = output_col_name + out_suffix

                ensemble_config.ensemble_scheduling.step.append(config_step)
                node_configs.append(node_config)

        # Write the ensemble config file
        ensemble_path = os.path.join(export_path, self.name)
        os.makedirs(ensemble_path, exist_ok=True)
        os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)

        with open(os.path.join(ensemble_path, "config.pbtxt"), "w") as o:
            text_format.PrintMessage(ensemble_config, o)

        return (ensemble_config, node_configs)


def _find_column_source(node, column_name):
    for upstream_node in node.parents_with_dependencies:
        if column_name in upstream_node.output_columns.names and hasattr(
            upstream_node.op, "export"
        ):
            return upstream_node

    for upstream_node in node.parents_with_dependencies:
        source = _find_column_source(upstream_node, column_name)
        if source:
            return source

    return None
