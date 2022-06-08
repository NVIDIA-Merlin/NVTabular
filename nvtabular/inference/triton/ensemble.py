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

import json
import os
import warnings
from shutil import copyfile, copytree

import numpy as np
import tritonclient.grpc.model_config_pb2 as model_config
from google.protobuf import text_format

from merlin.core.dispatch import is_string_dtype
from merlin.schema import Tags
from nvtabular import ColumnSelector


def export_tensorflow_ensemble(
    model,
    workflow,
    name,
    model_path,
    label_columns=None,
    sparse_max=None,
    version=1,
    nvtabular_backend="nvtabular",
    cats=None,
    conts=None,
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
    cats:
        Names of the categorical columns
    conts:
        Names of the continuous columns
    label_columns:
        Labels in the dataset (will be removed from the dataset)
    sparse_max:
        Max length of the each row when the sparse data is converted to dense
    version:
        Version of the model
    nvtabular_backend: "python" or "nvtabular"
        The backend that will be used for inference in Triton.
    """
    labels = (
        label_columns
        or workflow.output_schema.apply(ColumnSelector(tags=[Tags.TARGET])).column_names
    )
    workflow = workflow.remove_inputs(labels)

    # generate the TF saved model
    tf_path = os.path.join(model_path, name + "_tf")
    tf_config = export_tensorflow_model(model, name + "_tf", tf_path, version=version)

    # override the output dtype of the nvtabular model if necessary (fixes mismatches
    # in dtypes between tf inputs and nvt outputs)
    for column in tf_config.input:
        tf_dtype = _triton_datatype_to_dtype(column.data_type)
        nvt_col_name = column.name.replace("__values", "").replace("__nnzs", "")
        col_schema = workflow.output_schema[nvt_col_name]
        if col_schema.dtype and col_schema.dtype != tf_dtype:
            warnings.warn(
                f"TF model expects {tf_dtype} for column {col_schema.name}, but workflow "
                f" is producing type {col_schema.dtype}. Overriding dtype in NVTabular workflow."
            )
            workflow.output_schema.column_schemas[col_schema.name] = col_schema.with_dtype(tf_dtype)

    # generate the nvtabular triton model
    preprocessing_path = os.path.join(model_path, name + "_nvt")
    nvt_config = generate_nvtabular_model(
        workflow,
        name + "_nvt",
        preprocessing_path,
        sparse_max=sparse_max,
        backend=nvtabular_backend,
        cats=cats,
        conts=conts,
    )

    # generate the triton ensemble
    ensemble_path = os.path.join(model_path, name)
    os.makedirs(ensemble_path, exist_ok=True)
    os.makedirs(os.path.join(ensemble_path, str(version)), exist_ok=True)
    _generate_ensemble_config(name, ensemble_path, nvt_config, tf_config)


def export_pytorch_ensemble(
    model,
    workflow,
    sparse_max,
    name,
    model_path,
    label_columns=None,
    use_fix_dtypes=True,
    version=1,
    nvtabular_backend="python",
    cats=None,
    conts=None,
):
    """Creates an ensemble triton server model, with the first model being a nvtabular
    preprocessing, and the second by a pytorch savedmodel

    Parameters
    ----------
    model:
        The pytorch model that should be served
    workflow:
        The nvtabular workflow used in preprocessing
    sparse_max:
        Max length of the each row when the sparse data is converted to dense
    name:
        The base name of the various triton models
    model_path:
        The root path to write out files to
    cats:
        Names of the categorical columns
    conts:
        Names of the continuous columns
    label_columns:
        Labels in the dataset (will be removed from the dataset)
    use_fix_dtypes:
        Transformers4Rec is using fixed dtypes and this option is
        whether to use fixed dtypes in inference or not
    version:
        Version of the model
    nvtabular_backend: "python" or "nvtabular"
        The backend that will be used for inference in Triton.
    """
    labels = (
        label_columns
        or workflow.output_schema.apply(ColumnSelector(tags=[Tags.TARGET])).column_names
    )
    workflow = workflow.remove_inputs(labels)

    # generate the TF saved model
    pt_path = os.path.join(model_path, name + "_pt")
    pt_config = export_pytorch_model(
        model, workflow, sparse_max, name + "_pt", pt_path, use_fix_dtypes, version=version
    )

    # override the output dtype of the nvtabular model if necessary (fixes mismatches
    # in dtypes between tf inputs and nvt outputs)
    for column in pt_config.input:
        pt_dtype = _triton_datatype_to_dtype(column.data_type)
        nvt_dtype = workflow.output_dtypes.get(column.name)
        if nvt_dtype and nvt_dtype != pt_dtype:
            warnings.warn(
                f"PyTorch model expects {pt_dtype} for column {column.name}, but workflow "
                f" is producing type {nvt_dtype}. Overriding dtype in NVTabular workflow."
            )
            workflow.output_dtypes[column.name] = pt_dtype

    # generate the nvtabular triton model
    preprocessing_path = os.path.join(model_path, name + "_nvt")
    nvt_config = generate_nvtabular_model(
        workflow,
        name + "_nvt",
        preprocessing_path,
        backend=nvtabular_backend,
        cats=cats,
        conts=conts,
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
    version=1,
    max_batch_size=None,
    nvtabular_backend="python",
    cats=None,
    conts=None,
    label_columns=None,
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
    version:
        The version of the model
    max_batch_size:
        Max batch size that Triton can receive
    nvtabular_backend: "python" or "nvtabular"
        The backend that will be used for inference in Triton.
    cats:
        Names of the categorical columns
    conts:
        Names of the continuous columns
    label_columns:
        Labels in the dataset (will be removed from the dataset)
    """
    cats = cats or workflow.output_schema.apply(ColumnSelector(tags=[Tags.CATEGORICAL]))
    conts = conts or workflow.output_schema.apply(ColumnSelector(tags=[Tags.CONTINUOUS]))
    labels = label_columns or workflow.output_schema.apply(ColumnSelector(tags=[Tags.TARGET]))

    if not cats and not conts:
        raise ValueError("Either cats or conts has to have a value.")

    workflow = workflow.remove_inputs(labels)

    # generate the nvtabular triton model
    preprocessing_path = os.path.join(output_path, name + "_nvt")
    nvt_config = generate_nvtabular_model(
        workflow=workflow,
        name=name + "_nvt",
        output_path=preprocessing_path,
        version=version,
        output_model="hugectr",
        max_batch_size=max_batch_size,
        backend=nvtabular_backend,
        cats=cats,
        conts=conts,
    )

    hugectr_params["label_dim"] = len(labels)
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


def generate_nvtabular_model(
    workflow,
    name,
    output_path,
    version=1,
    output_model=None,
    max_batch_size=None,
    sparse_max=None,
    backend="python",
    cats=None,
    conts=None,
):
    """converts a workflow to a triton mode
    Parameters
    ----------
    sparse_max:
        Max length of the each row when the sparse data is converted to dense
    cats:
        Names of the categorical columns
    conts:
        Names of the continuous columns
    """
    workflow.save(os.path.join(output_path, str(version), "workflow"))
    config = _generate_nvtabular_config(
        workflow,
        name,
        output_path,
        output_model,
        max_batch_size,
        sparse_max=sparse_max,
        backend=backend,
        cats=cats,
        conts=conts,
    )

    # copy the model file over. note that this isn't necessary with the c++ backend, but
    # does provide us to use the python backend with just changing the 'backend' parameter
    copyfile(
        os.path.join(os.path.dirname(__file__), "workflow_model.py"),
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


def _generate_nvtabular_config(
    workflow,
    name,
    output_path,
    output_model=None,
    max_batch_size=None,
    sparse_max=None,
    backend="python",
    cats=None,
    conts=None,
):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""
    config = model_config.ModelConfig(name=name, backend=backend, max_batch_size=max_batch_size)

    config.parameters["python_module"].string_value = "nvtabular.inference.triton.workflow_model"
    config.parameters["output_model"].string_value = output_model if output_model else ""

    config.parameters["cats"].string_value = json.dumps(cats) if cats else ""
    config.parameters["conts"].string_value = json.dumps(conts) if conts else ""

    if sparse_max:
        # this assumes seq_length is same for each list column
        config.parameters["sparse_max"].string_value = json.dumps(sparse_max)

    if output_model == "hugectr":
        config.instance_group.append(model_config.ModelInstanceGroup(kind=2))

        for column in workflow.output_node.input_columns.names:
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
        for col_name, col_schema in workflow.input_schema.column_schemas.items():
            _add_model_param(col_schema, model_config.ModelInput, config.input)

        for col_name, col_schema in workflow.output_schema.column_schemas.items():
            _add_model_param(
                col_schema,
                model_config.ModelOutput,
                config.output,
                [-1, 1],
            )
    else:
        for col_name, col_schema in workflow.input_schema.column_schemas.items():
            _add_model_param(col_schema, model_config.ModelInput, config.input)

        for col_name, col_schema in workflow.output_schema.column_schemas.items():
            if sparse_max and col_name in sparse_max.keys():
                # this assumes max_sequence_length is equal for all output columns
                dim = sparse_max[col_name]
                _add_model_param(col_schema, model_config.ModelOutput, config.output, [-1, dim])
            else:
                _add_model_param(col_schema, model_config.ModelOutput, config.output)

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

    config.parameters["TF_GRAPH_TAG"].string_value = "serve"
    config.parameters["TF_SIGNATURE_DEF"].string_value = "serving_default"

    for col in inputs:
        config.input.append(
            model_config.ModelInput(
                name=col.name, data_type=_convert_dtype(col.dtype), dims=[-1, col.shape[1]]
            )
        )

    for col in outputs:
        # this assumes the list columns are 1D tensors both for cats and conts
        config.output.append(
            model_config.ModelOutput(
                name=col.name.split("/")[0],
                data_type=_convert_dtype(col.dtype),
                dims=[-1, col.shape[1]],
            )
        )

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


def export_pytorch_model(
    model, workflow, sparse_max, name, output_path, use_fix_dtypes=True, version=1, backend="python"
):
    """Exports a PyTorch model for serving with Triton

    Parameters
    ----------
    model:
        The PyTorch model that should be served
    workflow:
        The nvtabular workflow used in preprocessing
    sparse_max:
        Max length of the each row when the sparse data is converted to dense
    name:
        The name of the triton model to export
    output_path:
        The path to write the exported model to
    use_fix_dtypes:
        Transformers4Rec is using fixed dtypes and this option is
        whether to use fixed dtypes in inference or not
    version:
        Version of the model
    backend: "python" or "nvtabular"
        The backend that will be used for inference in Triton.
    """
    import cloudpickle
    import torch

    os.makedirs(os.path.join(output_path, str(version)), exist_ok=True)

    pt_model_path = os.path.join(output_path, str(version), "model.pth")
    torch.save(model.state_dict(), pt_model_path)

    pt_model_path = os.path.join(output_path, str(version), "model.pkl")
    with open(pt_model_path, "wb") as o:
        cloudpickle.dump(model, o)

    copyfile(
        os.path.join(os.path.dirname(__file__), "model", "model_pt.py"),
        os.path.join(output_path, str(version), "model.py"),
    )

    config = model_config.ModelConfig(name=name, backend=backend)

    for col_name, col_schema in workflow.output_schema.column_schemas.items():
        _add_model_param(col_schema, model_config.ModelInput, config.input)

    *_, last_layer = model.parameters()
    dims = last_layer.shape[0]
    dtype = last_layer.dtype
    config.output.append(
        model_config.ModelOutput(
            name="output", data_type=_convert_pytorch_dtype(dtype), dims=[-1, dims]
        )
    )

    if sparse_max:
        with open(os.path.join(output_path, str(version), "model_info.json"), "w") as o:
            model_info = dict()
            model_info["sparse_max"] = sparse_max
            model_info["use_fix_dtypes"] = use_fix_dtypes
            json.dump(model_info, o)

    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)
    return config


def _generate_pytorch_config(model, name, output_path, max_batch_size=None):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""
    config = model_config.ModelConfig(name=name, backend="python", max_batch_size=max_batch_size)

    for col in model.inputs:
        config.input.append(
            model_config.ModelInput(name=col.name, data_type=_convert_dtype(col.dtype), dims=[-1])
        )

    for col in model.outputs:
        config.output.append(
            model_config.ModelOutput(
                name=col.name.split("/")[0], data_type=_convert_dtype(col.dtype), dims=[-1]
            )
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


def _add_model_param(col_schema, paramclass, params, dims=None):
    dims = dims if dims is not None else [-1, 1]
    if col_schema.is_list and col_schema.is_ragged:
        params.append(
            paramclass(
                name=col_schema.name + "__values",
                data_type=_convert_dtype(col_schema.dtype),
                dims=dims,
            )
        )
        params.append(
            paramclass(
                name=col_schema.name + "__nnzs", data_type=model_config.TYPE_INT64, dims=dims
            )
        )
    else:
        params.append(
            paramclass(name=col_schema.name, data_type=_convert_dtype(col_schema.dtype), dims=dims)
        )


def _convert_dtype(dtype):
    """converts a dtype to the appropriate triton proto type"""

    if dtype and not isinstance(dtype, str):
        dtype_name = dtype.name if hasattr(dtype, "name") else dtype.__name__
    else:
        dtype_name = dtype

    dtypes = {
        "float64": model_config.TYPE_FP64,
        "float32": model_config.TYPE_FP32,
        "float16": model_config.TYPE_FP16,
        "int64": model_config.TYPE_INT64,
        "int32": model_config.TYPE_INT32,
        "int16": model_config.TYPE_INT16,
        "int8": model_config.TYPE_INT8,
        "uint64": model_config.TYPE_UINT64,
        "uint32": model_config.TYPE_UINT32,
        "uint16": model_config.TYPE_UINT16,
        "uint8": model_config.TYPE_UINT8,
        "bool": model_config.TYPE_BOOL,
    }

    if is_string_dtype(dtype):
        return model_config.TYPE_STRING
    elif dtype_name in dtypes:
        return dtypes[dtype_name]
    else:
        raise ValueError(f"Can't convert {dtype} to a Triton dtype")


def _convert_pytorch_dtype(dtype):
    """converts a dtype to the appropriate triton proto type"""

    import torch

    dtypes = {
        torch.float64: model_config.TYPE_FP64,
        torch.float32: model_config.TYPE_FP32,
        torch.float16: model_config.TYPE_FP16,
        torch.int64: model_config.TYPE_INT64,
        torch.int32: model_config.TYPE_INT32,
        torch.int16: model_config.TYPE_INT16,
        torch.int8: model_config.TYPE_INT8,
        torch.uint8: model_config.TYPE_UINT8,
        torch.bool: model_config.TYPE_BOOL,
    }

    if is_string_dtype(dtype):
        return model_config.TYPE_STRING
    elif dtype in dtypes:
        return dtypes[dtype]
    else:
        raise ValueError(f"Can't convert dtype {dtype})")


def _convert_string2pytorch_dtype(dtype):
    """converts a dtype to the appropriate torch type"""

    import torch

    if not isinstance(dtype, str):
        dtype_name = dtype.name
    else:
        dtype_name = dtype

    dtypes = {
        "TYPE_FP64": torch.float64,
        "TYPE_FP32": torch.float32,
        "TYPE_FP16": torch.float16,
        "TYPE_INT64": torch.int64,
        "TYPE_INT32": torch.int32,
        "TYPE_INT16": torch.int16,
        "TYPE_INT8": torch.int8,
        "TYPE_UINT8": torch.uint8,
        "TYPE_BOOL": torch.bool,
    }

    if is_string_dtype(dtype):
        return model_config.TYPE_STRING
    elif dtype_name in dtypes:
        return dtypes[dtype_name]
    else:
        raise ValueError(f"Can't convert dtype {dtype})")


def _triton_datatype_to_dtype(data_type):
    """the reverse of _convert_dtype: converts a triton proto data_type to a numpy dtype"""
    name = model_config._DATATYPE.values[data_type].name[5:].lower()
    if name == "string":
        return np.dtype("str")
    return np.dtype(name.replace("fp", "float"))
