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

import copy
import json
import os
import subprocess
from shutil import copyfile

import cudf
import tritonclient.grpc as grpcclient
from google.protobuf import text_format
from tritonclient.utils import np_to_triton_dtype

# read in the triton ModelConfig proto object - generating it if it doesn't exist
try:
    import nvtabular.inference.triton.model_config_pb2 as model_config
except ImportError:
    pwd = os.path.dirname(__file__)
    try:
        subprocess.check_output(
            ["protoc", f"--python_out={pwd}", f"--proto_path={pwd}", "model_config.proto"]
        )
    except Exception as e:
        raise ImportError("Failed to compile model_config.proto - is protobuf installed?") from e
    import nvtabular.inference.triton.model_config_pb2 as model_config


def export_tensorflow_ensemble(model, workflow, name, model_path, label_columns, version=1):
    """Creates an ensemble triton server model, with the first model being a nvtabular
    preprocessing, and the second by a tensorflow savedmodel

    Parameters
    ----------
    model:
        The tensorflow model that should be served
    workflow:
        The nvtabular workflow used in preprocessing
    name:
        The base name of the various triton models
    model_path:
        The root path to write out files to
    label_columns:
        Labels in the dataset (will be removed f
    """

    workflow = _remove_columns(workflow, label_columns)

    # generate the nvtabular triton model
    preprocessing_path = os.path.join(model_path, name + "_nvt")
    nvt_config = generate_triton_model(workflow, name + "_nvt", preprocessing_path)

    # generate the TF saved model
    tf_path = os.path.join(model_path, name + "_tf")
    tf_model_path = os.path.join(tf_path, str(version), "model.savedmodel")
    model.save(tf_model_path)
    tf_config = _generate_tensorflow_config(model, name + "_tf", tf_path)

    # generate the triton ensemble
    ensemble_path = os.path.join(model_path, name)
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)
    _generate_ensemble_config(model, workflow, name, ensemble_path, nvt_config, tf_config)


def _remove_columns(workflow, to_remove):
    workflow = copy.deepcopy(workflow)

    workflow.column_group = _remove_columns_from_column_group(workflow.column_group, to_remove)

    for label in to_remove:
        if label in workflow.input_dtypes:
            del workflow.input_dtypes[label]

        if label in workflow.output_dtypes:
            del workflow.output_dtypes[label]

    return workflow


def _remove_columns_from_column_group(cg, to_remove):
    cg.columns = [col for col in cg.columns if col not in to_remove]
    parents = [_remove_columns_from_column_group(parent, to_remove) for parent in cg.parents]
    cg.parents = [p for p in parents if p.columns]
    return cg


def _generate_ensemble_config(model, workflow, name, output_path, nvt_config, tf_config):
    config = model_config.ModelConfig(name=name, platform="ensemble")
    config.input.extend(nvt_config.input)
    config.output.extend(tf_config.output)

    nvt_step = model_config.ModelEnsembling.Step(model_name=name + "_nvt", model_version=-1)
    for input_col in nvt_config.input:
        nvt_step.input_map[input_col.name] = input_col.name
    for output_col in nvt_config.output:
        nvt_step.output_map[output_col.name] = output_col.name + "_nvt"

    tf_step = model_config.ModelEnsembling.Step(model_name=name + "_tf", model_version=-1)
    for input_col in tf_config.input:
        tf_step.input_map[input_col.name] = input_col.name + "_nvt"
    for output_col in tf_config.output:
        tf_step.output_map[output_col.name] = output_col.name

    config.ensemble_scheduling.step.append(nvt_step)
    config.ensemble_scheduling.step.append(tf_step)

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


def _generate_tensorflow_config(model, name, output_path):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""
    config = model_config.ModelConfig(
        name=name, backend="tensorflow", platform="tensorflow_savedmodel"
    )

    for col in model.inputs:
        config.input.append(
            model_config.ModelInput(
                name=col.name, data_type=_convert_dtype(col.dtype), dims=[-1, 1]
            )
        )

    for col in model.outputs:
        config.output.append(
            model_config.ModelOutput(
                name=col.name.split("/")[0], data_type=_convert_dtype(col.dtype), dims=[-1, 1]
            )
        )

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


def generate_triton_model(
    workflow,
    name,
    output_path,
    version=1,
    output_model=None,
    cats=None,
    conts=None,
    max_batch_size=None,
):
    """ converts a workflow to a triton mode """

    workflow.save(os.path.join(output_path, str(version), "workflow"))
    config = _generate_model_config(
        workflow, name, output_path, output_model, max_batch_size, cats, conts
    )

    if output_model == "hugectr":
        _generate_column_types(os.path.join(output_path, str(version), "workflow"), cats, conts)
        copyfile(
            os.path.join(os.path.dirname(__file__), "model_hugectr.py"),
            os.path.join(output_path, str(version), "model.py"),
        )
    else:
        copyfile(
            os.path.join(os.path.dirname(__file__), "model.py"),
            os.path.join(output_path, str(version), "model.py"),
        )

    return config


def convert_df_to_triton_input(column_names, batch, input_class=grpcclient.InferInput):
    columns = [(col, batch[col]) for col in column_names]
    inputs = [input_class(name, col.shape, np_to_triton_dtype(col.dtype)) for name, col in columns]
    for i, (name, col) in enumerate(columns):
        inputs[i].set_data_from_numpy(col.values_host)
    return inputs


def convert_triton_output_to_df(columns, response):
    return cudf.DataFrame({col: response.as_numpy(col) for col in columns})


def _generate_model_config(
    workflow, name, output_path, output_model=None, max_batch_size=None, cats=None, conts=None
):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""

    config = model_config.ModelConfig(name=name, backend="python", max_batch_size=max_batch_size)

    if output_model == "hugectr":
        for column in workflow.column_group.input_column_names:
            dtype = workflow.input_dtypes[column]
            config.input.append(
                model_config.ModelInput(name=column, data_type=_convert_dtype(dtype), dims=[-1])
            )

        config.output.append(
            model_config.ModelOutput(name="DES", data_type=model_config.TYPE_FP32, dims=[-1])
        )

        config.output.append(
            model_config.ModelOutput(
                name="CATCOLUMN", data_type=model_config.TYPE_UINT32, dims=[-1]
            )
        )

        config.output.append(
            model_config.ModelOutput(name="ROWINDEX", data_type=model_config.TYPE_INT32, dims=[-1])
        )
    else:
        for column, dtype in workflow.input_dtypes.items():
            config.input.append(
                model_config.ModelInput(name=column, data_type=_convert_dtype(dtype), dims=[-1, 1])
            )

        for column, dtype in workflow.output_dtypes.items():
            config.output.append(
                model_config.ModelOutput(name=column, data_type=_convert_dtype(dtype), dims=[-1, 1])
            )

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


def _generate_column_types(output_path, cats=None, conts=None):
    if cats is None and conts is None:
        raise ValueError("Either cats or conts has to have a value.")

    if cats or conts:
        with open(os.path.join(output_path, "column_types.json"), "w") as o:
            cats_conts_json = dict()
            if cats:
                cats_conts_json["cats"] = [name for i, name in enumerate(cats)]
            if conts:
                cats_conts_json["conts"] = [name for i, name in enumerate(conts)]
            json.dump(cats_conts_json, o)


def get_column_types(path):
    return json.load(open(os.path.join(path, "column_types.json")))


def _convert_dtype(dtype):
    """ converts a dtype to the appropriate triton proto type """
    if dtype == "float64":
        return model_config.TYPE_FP64
    if dtype == "float32":
        return model_config.TYPE_FP32
    if dtype == "float16":
        return model_config.TYPE_FP16
    if dtype == "int64":
        return model_config.TYPE_INT64
    if dtype == "int32":
        return model_config.TYPE_INT32
    if dtype == "int16":
        return model_config.TYPE_INT16
    if dtype == "int8":
        return model_config.TYPE_INT8
    if dtype == "uint64":
        return model_config.TYPE_UINT64
    if dtype == "uint32":
        return model_config.TYPE_UINT32
    if dtype == "uint16":
        return model_config.TYPE_UINT16
    if dtype == "uint8":
        return model_config.TYPE_UINT8
    if dtype == "bool":
        return model_config.TYPE_BOOL
    if cudf.utils.dtypes.is_string_dtype(dtype):
        return model_config.TYPE_STRING
    raise ValueError(f"Can't convert dtype {dtype})")
