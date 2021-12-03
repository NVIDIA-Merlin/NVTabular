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
import importlib
import json


class OperatorRunner:
    def __init__(self, repository, version, kind, config):
        operator_names = self.fetch_json_param(config, "operator_names")
        op_configs = [self.fetch_json_param(config, op_name) for op_name in operator_names]

        self.operators = []
        for op_config in op_configs:
            module_name = op_config["module_name"]
            class_name = op_config["class_name"]

            op_module = importlib.import_module(module_name)
            op_class = getattr(op_module, class_name)

            operator = op_class.from_config(op_config)
            self.operators.append(operator)

    def execute(self, tensors):
        for operator in self.operators:
            tensors = operator.transform(tensors)
        return tensors

    def fetch_json_param(self, model_config, param_name):
        string_value = model_config["parameters"][param_name]["string_value"]
        return json.loads(string_value)
