import json
import os
import pathlib
from abc import abstractclassmethod, abstractmethod
from shutil import copyfile

from merlin.dag import BaseOperator

import nvtabular as nvt

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from google.protobuf import text_format  # noqa

import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa
from nvtabular.inference.triton.ensemble import _convert_dtype  # noqa


class InferenceDataFrame:
    def __init__(self, tensors=None):
        self.tensors = tensors or {}

    def __getitem__(self, col_items):
        if isinstance(col_items, list):
            results = {name: self.tensors[name] for name in col_items}
            return InferenceDataFrame(results)
        else:
            return self.tensors[col_items]

    def __len__(self):
        return len(self.tensors)

    def __iter__(self):
        for name, tensor in self.tensors.items():
            yield name, tensor

    def __repr__(self):
        dict_rep = {}
        for k, v in self.tensors.items():
            dict_rep[k] = v
        return str(dict_rep)


class InferenceOperator(BaseOperator):
    @property
    def export_name(self):
        return self.__class__.__name__.lower()

    @abstractmethod
    def export(self, export_path, input_schema, output_schema, node_id=None, version=1):
        pass

    def create_node(self, selector):
        return nvt.inference.graph.node.InferenceNode(selector)


class PipelineableInferenceOperator(InferenceOperator):
    @abstractclassmethod
    def from_config(cls, config):
        pass

    @abstractmethod
    def transform(self, df: InferenceDataFrame) -> InferenceDataFrame:
        """Transform the dataframe by applying this operator to the set of input columns

        Parameters
        -----------
        df: Dataframe
            A pandas or cudf dataframe that this operator will work on

        Returns
        -------
        DataFrame
            Returns a transformed dataframe for this operator
        """

    def export(self, path, input_schema, output_schema, node_id=None, version=1):
        node_name = f"{self.export_name}_{node_id}" if node_id is not None else self.export_name

        node_export_path = pathlib.Path(path) / node_name
        node_export_path.mkdir(exist_ok=True)

        config = model_config.ModelConfig(name=node_name, backend="nvtabular", platform="op_runner")

        config.parameters["operator_names"].string_value = json.dumps([node_name])

        config.parameters[node_name].string_value = json.dumps(
            {
                "module_name": self.__class__.__module__,
                "class_name": self.__class__.__name__,
                "input_dict": json.dumps(_schema_to_dict(input_schema)),
                "output_dict": json.dumps(_schema_to_dict(output_schema)),
            }
        )

        for col_name, col_dict in _schema_to_dict(input_schema).items():
            config.input.append(
                model_config.ModelInput(
                    name=col_name, data_type=_convert_dtype(col_dict["dtype"]), dims=[-1, 1]
                )
            )

        for col_name, col_dict in _schema_to_dict(output_schema).items():
            # this assumes the list columns are 1D tensors both for cats and conts
            config.output.append(
                model_config.ModelOutput(
                    name=col_name.split("/")[0],
                    data_type=_convert_dtype(col_dict["dtype"]),
                    dims=[-1, 1],
                )
            )

        with open(os.path.join(node_export_path, "config.pbtxt"), "w") as o:
            text_format.PrintMessage(config, o)

        os.makedirs(node_export_path, exist_ok=True)
        os.makedirs(os.path.join(node_export_path, str(version)), exist_ok=True)
        copyfile(
            os.path.join(os.path.dirname(__file__), "..", "..", "triton", "oprunner_model.py"),
            os.path.join(node_export_path, str(version), "model.py"),
        )

        return config


def _schema_to_dict(schema):
    # TODO: Write the conversion
    schema_dict = {}
    for col_name, col_schema in schema.column_schemas.items():
        schema_dict[col_name] = {
            "dtype": col_schema.dtype.name,
            "is_list": col_schema.is_list,
        }

    return schema_dict
