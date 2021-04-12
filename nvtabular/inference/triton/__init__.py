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
from cudf.utils.dtypes import is_list_dtype
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
    nvt_config = generate_nvtabular_model(workflow, name + "_nvt", preprocessing_path)

    # generate the TF saved model
    tf_path = os.path.join(model_path, name + "_tf")
    tf_model_path = os.path.join(tf_path, str(version), "model.savedmodel")
    model.save(tf_model_path)
    tf_config = _generate_tensorflow_config(model, name + "_tf", tf_path)

    # generate the triton ensemble
    ensemble_path = os.path.join(model_path, name)
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)
    _generate_ensemble_config(name, ensemble_path, nvt_config, tf_config)


def export_pytorch_ensemble(
    model, model_info, sample_input_data, workflow, name, model_path, label_columns, version=1
):
    """Creates an ensemble triton server model, with the first model being a nvtabular
    preprocessing, and the second by a pytorch saved model

    Parameters
    ----------
    model:
        The pytorch model that should be served
    workflow:
        The nvtabular workflow used in preprocessing
    name:
        The base name of the various triton models
    model_path:
        The root path to write out files to
    label_columns:
        Labels in the dataset (will be removed f
    """
    import torch

    workflow = _remove_columns(workflow, label_columns)

    # generate the nvtabular triton model
    preprocessing_path = os.path.join(model_path, name + "_nvt")
    nvt_config = generate_nvtabular_model(
        workflow=workflow,
        name=name + "_nvt",
        output_path=preprocessing_path,
        version=version,
        output_model="pytorch",
        output_info=model_info["input"],
    )

    dynamic_axes = dict()
    model_input_names = []
    for model_input_name in model_info["input"]:
        model_input_names.append(model_input_name)
        dynamic_axes[model_input_name] = {0: "batch_size"}

    model_output_names = []
    for model_output_name in model_info["output"]:
        model_output_names.append(model_output_name)
        dynamic_axes[model_output_name] = {0: "batch_size"}

    # generate the PT saved model
    pt_path = os.path.join(model_path, name + "_pt")
    pt_model_path = os.path.join(pt_path, str(version), "model.onnx")

    os.makedirs(pt_path, exist_ok=True)
    os.makedirs(os.path.join(pt_path, str(version)), exist_ok=True)

    pt_config = _generate_pytorch_config(name + "_pt", pt_path, model_info)

    torch.onnx.export(
        model,
        sample_input_data,
        pt_model_path,
        export_params=True,
        input_names=model_input_names,  # the model's input names
        output_names=model_output_names,
        dynamic_axes=dynamic_axes,
    )

    # generate the triton ensemble
    ensemble_path = os.path.join(model_path, name)
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)
    _generate_ensemble_config(name, ensemble_path, nvt_config, pt_config)


def export_hugectr_ensemble(
    workflow,
    hugectr_model_path,
    hugectr_params,
    name,
    output_path,
    label_columns,
    version=1,
    cats=None,
    conts=None,
    max_batch_size=None,
):
    """Creates an ensemble hugectr server model, with the first model being a nvtabular
    preprocessing, and the second by a hugectr savedmodel

    Parameters
    ----------
    workflow:
        The nvtabular workflow used in preprocessing
    hugectr_model_path:
        The path of the trained model files
    hugectr_params:
        HugeCTR specific parameters
    name:
        The base name of the various triton models
    output_path:
        The path where the models will be served
    label_columns:
        Labels in the dataset (will be removed from the workflow)
    version:
        The version of the model
    cats:
        Names of the categorical columns
    conts:
        Names of the continous columns
    max_batch_size:
        Max batch size that Triton can receive

    """

    workflow = _remove_columns(workflow, label_columns)

    # generate the nvtabular triton model
    preprocessing_path = os.path.join(output_path, name + "_nvt")
    nvt_config = generate_nvtabular_model(
        workflow=workflow,
        name=name + "_nvt",
        output_path=preprocessing_path,
        version=version,
        output_model="hugectr",
        cats=cats,
        conts=conts,
        max_batch_size=max_batch_size,
    )

    hugectr_params["label_dim"] = len(label_columns)
    if conts is None:
        hugectr_params["des_feature_num"] = 0
    else:
        hugectr_params["des_feature_num"] = len(conts)

    if cats is None:
        hugectr_params["cat_feature_num"] = 0
    else:
        hugectr_params["cat_feature_num"] = len(cats)

    # generate the HugeCTR saved model
    hugectr_config = generate_hugectr_model(
        trained_model_path=hugectr_model_path,
        hugectr_params=hugectr_params,
        name=name,
        output_path=output_path,
        version=version,
        max_batch_size=max_batch_size,
    )

    hugectr_training_config = json.load(open(hugectr_params["config"]))
    for elem in hugectr_training_config["layers"]:
        if "slot_size_array" in elem:
            with open(
                os.path.join(preprocessing_path, str(version), "workflow", "slot_size_array.json"),
                "w",
            ) as o:
                slot_sizes = dict()
                slot_sizes["slot_size_array"] = elem["slot_size_array"]
                json.dump(slot_sizes, o)
            break

    # generate the triton ensemble
    ensemble_path = os.path.join(output_path, name + "_ens")
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)
    _generate_ensemble_config(name, ensemble_path, nvt_config, hugectr_config, "_ens")


def generate_nvtabular_model(
    workflow,
    name,
    output_path,
    version=1,
    output_model=None,
    cats=None,
    conts=None,
    max_batch_size=None,
    output_info=None,
):
    """ converts a workflow to a triton mode """

    workflow.save(os.path.join(output_path, str(version), "workflow"))
    config = _generate_nvtabular_config(
        workflow, name, output_path, output_model, max_batch_size, cats, conts, output_info
    )

    if output_model == "hugectr":
        _generate_column_types(os.path.join(output_path, str(version), "workflow"), cats, conts)
        copyfile(
            os.path.join(os.path.dirname(__file__), "model_hugectr.py"),
            os.path.join(output_path, str(version), "model.py"),
        )
    elif output_model == "pytorch":
        _generate_column_types_pytorch(
            os.path.join(output_path, str(version), "workflow"), output_info=output_info
        )
        copyfile(
            os.path.join(os.path.dirname(__file__), "model_pytorch.py"),
            os.path.join(output_path, str(version), "model.py"),
        )
    else:
        copyfile(
            os.path.join(os.path.dirname(__file__), "model.py"),
            os.path.join(output_path, str(version), "model.py"),
        )

    return config


def generate_hugectr_model(
    trained_model_path,
    hugectr_params,
    name,
    output_path,
    version=1,
    max_batch_size=None,
):
    """ converts a trained HugeCTR model to a triton mode """

    out_path = os.path.join(output_path, name)
    os.makedirs(os.path.join(output_path, name), exist_ok=True)
    out_path_version = os.path.join(out_path, str(version))
    os.makedirs(out_path_version, exist_ok=True)

    config = _generate_hugectr_config(name, out_path, hugectr_params, max_batch_size=max_batch_size)
    for fname in os.listdir(trained_model_path):
        copyfile(
            os.path.join(trained_model_path, fname),
            os.path.join(out_path_version, fname),
        )

    return config


def convert_df_to_triton_input(column_names, batch, input_class=grpcclient.InferInput):
    columns = [(col, batch[col]) for col in column_names]
    inputs = []
    for i, (name, col) in enumerate(columns):
        if is_list_dtype(col):
            inputs.append(
                _convert_column_to_triton_input(
                    col._column.offsets.values_host.astype("int64"), name + "__nnzs", input_class
                )
            )
            inputs.append(
                _convert_column_to_triton_input(
                    col.list.leaves.values_host.astype("int64"), name + "__values", input_class
                )
            )
        else:
            inputs.append(_convert_column_to_triton_input(col.values_host, name, input_class))
    return inputs


def _convert_column_to_triton_input(col, name, input_class=grpcclient.InferInput):
    col = col.reshape(len(col), 1)
    input_tensor = input_class(name, col.shape, np_to_triton_dtype(col.dtype))
    input_tensor.set_data_from_numpy(col)
    return input_tensor


def convert_triton_output_to_df(columns, response):
    return cudf.DataFrame({col: response.as_numpy(col) for col in columns})


def _generate_nvtabular_config(
    workflow,
    name,
    output_path,
    output_model=None,
    max_batch_size=None,
    cats=None,
    conts=None,
    output_info=None,
):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""

    config = model_config.ModelConfig(name=name, backend="python", max_batch_size=max_batch_size)

    if output_model == "hugectr":
        config.instance_group.append(model_config.ModelInstanceGroup(kind=2))

        for column in workflow.column_group.input_column_names:
            dtype = workflow.input_dtypes[column]
            config.input.append(
                model_config.ModelInput(name=column, data_type=_convert_dtype(dtype), dims=[-1])
            )

        config.output.append(
            model_config.ModelOutput(name="DES", data_type=model_config.TYPE_FP32, dims=[-1])
        )

        config.output.append(
            model_config.ModelOutput(name="CATCOLUMN", data_type=model_config.TYPE_INT64, dims=[-1])
        )

        config.output.append(
            model_config.ModelOutput(name="ROWINDEX", data_type=model_config.TYPE_INT32, dims=[-1])
        )
    elif output_model == "pytorch":
        for column, dtype in workflow.input_dtypes.items():
            _add_model_param(column, dtype, model_config.ModelInput, config.input)

        for col, val in output_info.items():
            _add_model_param(
                col,
                val["dtype"],
                model_config.ModelOutput,
                config.output,
                [-1, len(val["columns"])],
            )
    else:
        for column, dtype in workflow.input_dtypes.items():
            _add_model_param(column, dtype, model_config.ModelInput, config.input)

        for column, dtype in workflow.output_dtypes.items():
            _add_model_param(column, dtype, model_config.ModelOutput, config.output)

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


def _generate_ensemble_config(name, output_path, nvt_config, nn_config, name_ext=""):
    config = model_config.ModelConfig(
        name=name + name_ext, platform="ensemble", max_batch_size=nvt_config.max_batch_size
    )
    config.input.extend(nvt_config.input)
    config.output.extend(nn_config.output)

    nvt_step = model_config.ModelEnsembling.Step(model_name=nvt_config.name, model_version=-1)
    for input_col in nvt_config.input:
        nvt_step.input_map[input_col.name] = input_col.name
    for output_col in nvt_config.output:
        nvt_step.output_map[output_col.name] = output_col.name + "_nvt"

    tf_step = model_config.ModelEnsembling.Step(model_name=nn_config.name, model_version=-1)
    for input_col in nn_config.input:
        tf_step.input_map[input_col.name] = input_col.name + "_nvt"
    for output_col in nn_config.output:
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


def _generate_pytorch_config(name, output_path, model_info, max_batch_size=None):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""
    config = model_config.ModelConfig(
        name=name, platform="onnxruntime_onnx", max_batch_size=max_batch_size
    )

    for col, val in model_info["input"].items():
        config.input.append(
            model_config.ModelInput(
                name=col, data_type=_convert_dtype(val["dtype"]), dims=[-1, len(val["columns"])]
            )
        )

    for col, val in model_info["output"].items():
        if len(val["columns"]) == 1:
            dims = [-1]
        else:
            dims = [-1, len(val["columns"])]
        config.output.append(
            model_config.ModelOutput(name=col, data_type=_convert_dtype(val["dtype"]), dims=dims)
        )

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


def _generate_hugectr_config(name, output_path, hugectr_params, max_batch_size=None):
    config = model_config.ModelConfig(name=name, backend="hugectr", max_batch_size=max_batch_size)

    config.input.append(
        model_config.ModelInput(name="DES", data_type=model_config.TYPE_FP32, dims=[-1])
    )

    config.input.append(
        model_config.ModelInput(name="CATCOLUMN", data_type=model_config.TYPE_INT64, dims=[-1])
    )

    config.input.append(
        model_config.ModelInput(name="ROWINDEX", data_type=model_config.TYPE_INT32, dims=[-1])
    )

    for i in range(hugectr_params["n_outputs"]):
        config.output.append(
            model_config.ModelOutput(
                name="OUTPUT" + str(i), data_type=model_config.TYPE_FP32, dims=[-1]
            )
        )

    config.instance_group.append(model_config.ModelInstanceGroup(gpus=[0], count=1, kind=1))

    config_hugectr = model_config.ModelParameter(string_value=hugectr_params["config"])
    config.parameters["config"].CopyFrom(config_hugectr)

    gpucache_val = hugectr_params.get("gpucache", "true")

    gpucache = model_config.ModelParameter(string_value=gpucache_val)
    config.parameters["gpucache"].CopyFrom(gpucache)

    gpucacheper_val = str(hugectr_params.get("gpucacheper_val", "0.5"))

    gpucacheper = model_config.ModelParameter(string_value=gpucacheper_val)
    config.parameters["gpucacheper"].CopyFrom(gpucacheper)

    label_dim = model_config.ModelParameter(string_value=str(hugectr_params["label_dim"]))
    config.parameters["label_dim"].CopyFrom(label_dim)

    slots = model_config.ModelParameter(string_value=str(hugectr_params["slots"]))
    config.parameters["slots"].CopyFrom(slots)

    des_feature_num = model_config.ModelParameter(
        string_value=str(hugectr_params["des_feature_num"])
    )
    config.parameters["des_feature_num"].CopyFrom(des_feature_num)

    cat_feature_num = model_config.ModelParameter(
        string_value=str(hugectr_params["cat_feature_num"])
    )
    config.parameters["cat_feature_num"].CopyFrom(cat_feature_num)

    max_nnz = model_config.ModelParameter(string_value=str(hugectr_params["max_nnz"]))
    config.parameters["max_nnz"].CopyFrom(max_nnz)

    embedding_vector_size = model_config.ModelParameter(
        string_value=str(hugectr_params["embedding_vector_size"])
    )
    config.parameters["embedding_vector_size"].CopyFrom(embedding_vector_size)

    embeddingkey_long_type_val = hugectr_params.get("embeddingkey_long_type", "true")

    embeddingkey_long_type = model_config.ModelParameter(string_value=embeddingkey_long_type_val)
    config.parameters["embeddingkey_long_type"].CopyFrom(embeddingkey_long_type)

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


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


def _add_model_param(column, dtype, paramclass, params, dims=[-1, 1]):
    if is_list_dtype(dtype):
        params.append(
            paramclass(
                name=column + "__values", data_type=_convert_dtype(dtype.element_type), dims=dims
            )
        )
        params.append(
            paramclass(name=column + "__nnzs", data_type=model_config.TYPE_INT64, dims=dims)
        )
    else:
        params.append(paramclass(name=column, data_type=_convert_dtype(dtype), dims=dims))


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


def _generate_column_types_pytorch(output_path, output_info):
    with open(os.path.join(output_path, "column_types.json"), "w") as o:
        json.dump(output_info, o)


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
