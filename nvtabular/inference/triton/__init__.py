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
import warnings
from shutil import copyfile, copytree

import numpy as np
import pandas as pd

# this needs to be before any modules that import protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tritonclient.grpc as grpcclient  # noqa
from google.protobuf import text_format  # noqa
from tritonclient.utils import np_to_triton_dtype  # noqa

import nvtabular.inference.triton.model_config_pb2 as model_config  # noqa
from nvtabular.dispatch import _is_list_dtype, _is_string_dtype, _make_df  # noqa
from nvtabular.ops import get_embedding_sizes  # noqa


def export_tensorflow_ensemble(
    model,
    workflow,
    name,
    model_path,
    label_columns,
    version=1,
    nvtabular_backend="python",
):
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

        Labels in the dataset (will be removed from the dataset)
    version:
        Version of the model
    nvtabular_backend: "python" or "nvtabular"
        The backend that will be used for inference in Triton.
    """

    workflow = _remove_columns(workflow, label_columns)

    # generate the TF saved model
    tf_path = os.path.join(model_path, name + "_tf")
    tf_config = export_tensorflow_model(model, name + "_tf", tf_path, version=version)

    # override the output dtype of the nvtabular model if necessary (fixes mismatches
    # in dtypes between tf inputs and nvt outputs)
    for column in tf_config.input:
        tf_dtype = _triton_datatype_to_dtype(column.data_type)
        nvt_dtype = workflow.output_dtypes.get(column.name)
        if nvt_dtype and nvt_dtype != tf_dtype:
            warnings.warn(
                f"TF model expects {tf_dtype} for column {column.name}, but workflow "
                f" is producing type {nvt_dtype}. Overriding dtype in NVTabular workflow."
            )
            workflow.output_dtypes[column.name] = tf_dtype

    # generate the nvtabular triton model
    preprocessing_path = os.path.join(model_path, name + "_nvt")
    nvt_config = generate_nvtabular_model(
        workflow,
        name + "_nvt",
        preprocessing_path,
        backend=nvtabular_backend,
    )

    # generate the triton ensemble
    ensemble_path = os.path.join(model_path, name)
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)
    _generate_ensemble_config(name, ensemble_path, nvt_config, tf_config)


def export_pytorch_ensemble(
    model,
    model_info,
    sample_input_data,
    workflow,
    name,
    model_path,
    label_columns,
    version=1,
    nvtabular_backend="python",
):
    """Creates an ensemble triton server model, with the first model being a nvtabular
    preprocessing, and the second by a pytorch saved model

    Parameters
    ----------
    model:
        The pytorch model that should be served
    model_info:
        Extra info about the model such as input dtype and shape
    sample_input_data:
        Sample input data to use it for converting PyTorch model
        to Onnx model
    workflow:
        The nvtabular workflow used in preprocessing
    name:
        The base name of the various triton models
    model_path:
        The root path to write out files to
    label_columns:
        Labels in the dataset (will be removed from the dataset)
    version:
        Version of the model
    nvtabular_backend: "python" or "nvtabular"
        The backend that will be used for inference in Triton.
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
        backend=nvtabular_backend,
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
    nvtabular_backend="python",
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
    nvtabular_backend: "python" or "nvtabular"
        The backend that will be used for inference in Triton.
    """

    if not cats and not conts:
        raise ValueError("Either cats or conts has to have a value.")

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
        backend=nvtabular_backend,
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
    backend="python",
):
    """converts a workflow to a triton mode"""

    workflow.save(os.path.join(output_path, str(version), "workflow"))
    config = _generate_nvtabular_config(
        workflow,
        name,
        output_path,
        output_model,
        max_batch_size,
        cats,
        conts,
        output_info,
        backend=backend,
    )

    if output_model == "hugectr":
        _generate_column_types(os.path.join(output_path, str(version), "workflow"), cats, conts)
    elif output_model == "pytorch":
        _generate_column_types_pytorch(
            os.path.join(output_path, str(version), "workflow"), output_info=output_info
        )

    # copy the model file over. note that this isn't necessary with the c++ backend, but
    # does provide us to use the python backend with just changing the 'backend' parameter
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
    """converts a trained HugeCTR model to a triton mode"""

    out_path = os.path.join(output_path, name)
    os.makedirs(os.path.join(output_path, name), exist_ok=True)
    out_path_version = os.path.join(out_path, str(version))
    os.makedirs(out_path_version, exist_ok=True)

    config = _generate_hugectr_config(name, out_path, hugectr_params, max_batch_size=max_batch_size)
    copytree(trained_model_path, out_path_version, dirs_exist_ok=True)

    return config


def convert_df_to_triton_input(column_names, batch, input_class=grpcclient.InferInput):
    columns = [(col, batch[col]) for col in column_names]
    inputs = []
    for i, (name, col) in enumerate(columns):
        if _is_list_dtype(col):
            if isinstance(col, pd.Series):
                raise ValueError("this function doesn't support CPU list values yet")
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
            values = col.values if isinstance(col, pd.Series) else col.values_host
            inputs.append(_convert_column_to_triton_input(values, name, input_class))
    return inputs


def _convert_column_to_triton_input(col, name, input_class=grpcclient.InferInput):
    col = col.reshape(len(col), 1)
    input_tensor = input_class(name, col.shape, np_to_triton_dtype(col.dtype))
    input_tensor.set_data_from_numpy(col)
    return input_tensor


def convert_triton_output_to_df(columns, response):
    return _make_df({col: response.as_numpy(col) for col in columns})


def _generate_nvtabular_config(
    workflow,
    name,
    output_path,
    output_model=None,
    max_batch_size=None,
    cats=None,
    conts=None,
    output_info=None,
    backend="python",
):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""
    config = model_config.ModelConfig(name=name, backend=backend, max_batch_size=max_batch_size)

    config.parameters["python_module"].string_value = "nvtabular.inference.triton.model"
    config.parameters["output_model"].string_value = output_model if output_model else ""

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

    nn_input_cols = set(col.name for col in nn_config.input)

    nvt_step = model_config.ModelEnsembling.Step(model_name=nvt_config.name, model_version=-1)
    for input_col in nvt_config.input:
        nvt_step.input_map[input_col.name] = input_col.name
    for output_col in nvt_config.output:
        if output_col.name not in nn_input_cols:
            warnings.warn(
                f"Column {output_col.name} is being generated by NVTabular workflow "
                f" but is unused in {nn_config.name} model"
            )
            continue
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


def export_tensorflow_model(model, name, output_path, version=1):
    """Exports a TensorFlow model for serving with Triton

    Parameters
    ----------
    model:
        The tensorflow model that should be served
    name:
        The name of the triton model to export
    output_path:
        The path to write the exported model to
    """
    tf_model_path = os.path.join(output_path, str(version), "model.savedmodel")
    model.save(tf_model_path, include_optimizer=False)
    config = model_config.ModelConfig(
        name=name, backend="tensorflow", platform="tensorflow_savedmodel"
    )

    inputs, outputs = model.inputs, model.outputs

    if not inputs or not outputs:
        signatures = getattr(model, "signatures", {}) or {}
        default_signature = signatures.get("serving_default")
        if not default_signature:
            # roundtrip saved model to disk to generate signature if it doesn't exist
            import tensorflow as tf

            reloaded = tf.keras.models.load_model(tf_model_path)
            default_signature = reloaded.signatures["serving_default"]

        inputs = list(default_signature.structured_input_signature[1].values())
        outputs = list(default_signature.structured_outputs.values())

    for col in inputs:
        config.input.append(
            model_config.ModelInput(
                name=col.name, data_type=_convert_dtype(col.dtype), dims=[-1, 1]
            )
        )

    for col in outputs:
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


def _add_model_param(column, dtype, paramclass, params, dims=None):
    dims = dims if dims is not None else [-1, 1]
    if _is_list_dtype(dtype):
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
    """converts a dtype to the appropriate triton proto type"""
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
    if _is_string_dtype(dtype):
        return model_config.TYPE_STRING
    raise ValueError(f"Can't convert dtype {dtype})")


def _triton_datatype_to_dtype(data_type):
    """the reverse of _convert_dtype: converts a triton proto data_type to a numpy dtype"""
    name = model_config._DATATYPE.values[data_type].name[5:].lower()
    if name == "string":
        return np.dtype("str")
    return np.dtype(name.replace("fp", "float"))


def _convert_tensor(t):
    out = t.as_numpy()
    if len(out.shape) == 2:
        out = out[:, 0]
    # cudf doesn't seem to handle dtypes like |S15 or object that well
    if _is_string_dtype(out.dtype):
        out = out.astype("str")
    return out
