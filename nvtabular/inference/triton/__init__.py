import os
import subprocess
from shutil import copyfile
import json

import cudf
import tritonclient.http as httpclient
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


def generate_triton_model(workflow, name, output_path, version=1, output_model=None, cats=None, conts=None, max_batch_size=None):
    """ converts a workflow to a triton mode """
    workflow.save(os.path.join(output_path, str(version), "workflow"))
    _generate_model_config(workflow, name, output_path, output_model, max_batch_size, cats, conts)

    if output_model is None:
        copyfile(
            os.path.join(os.path.dirname(__file__), "model.py"),
            os.path.join(output_path, str(version), "model.py"),
        )
    elif output_model == "hugectr":
        _generate_column_types(os.path.join(output_path, str(version), "workflow"), cats, conts)
        copyfile(
            os.path.join(os.path.dirname(__file__), "model_hugectr.py"),
            os.path.join(output_path, str(version), "model.py"),
        )

def _generate_column_types(output_path, cats=None, conts=None):
    if cats is None and conts is None:
        raise ValueError('Either cats or conts has to have a value.')

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

def convert_df_to_triton_input(column_names, batch, input_class=httpclient.InferInput):
    columns = [(col, batch[col]) for col in column_names]
    inputs = [input_class(name, col.shape, np_to_triton_dtype(col.dtype)) for name, col in columns]
    for i, (name, col) in enumerate(columns):
        inputs[i].set_data_from_numpy(col.values_host)
    return inputs


def convert_triton_output_to_df(columns, response):
    return cudf.DataFrame({col: response.as_numpy(col) for col in columns})


def _generate_model_config(workflow, name, output_path, output_model=None, max_batch_size=None, cats=None, conts=None):
    """given a workflow generates the trton modelconfig proto object describing the inputs
    and outputs to that workflow"""
    if max_batch_size is None:
        config = model_config.ModelConfig(name=name, backend="python")
    else:
        config = model_config.ModelConfig(name=name, backend="python", max_batch_size=max_batch_size)

    if output_model is None:
        for column in workflow.column_group.input_column_names:
            dtype = workflow.input_dtypes[column]
            config.input.append(
                model_config.ModelInput(name=column, data_type=_convert_dtype(dtype), dims=[-1])
            )

        for column, dtype in workflow.output_dtypes.items():
            config.output.append(
                model_config.ModelOutput(name=column, data_type=_convert_dtype(dtype), dims=[-1])
            )
    elif output_model == "hugectr":
        for column in workflow.column_group.input_column_names:
            dtype = workflow.input_dtypes[column]
            config.input.append(
                model_config.ModelInput(name=column, data_type=_convert_dtype(dtype), dims=[-1])
            )

        if conts:
            config.output.append(
                model_config.ModelOutput(name="DES", data_type=model_config.TYPE_FP32, dims=[-1])
            )

        if cats:
            config.output.append(
                model_config.ModelOutput(name="CATCOLUMN", data_type=model_config.TYPE_UINT32, dims=[-1])
            )

        config.output.append(
            model_config.ModelOutput(name="ROWINDEX", data_type=model_config.TYPE_INT32, dims=[-1])
        )


    with open(os.path.join(output_path, "config.pbtxt"), "w") as o:
        text_format.PrintMessage(config, o)


def _convert_dtype(dtype):
    """ converts a dtype to the appropiate triton proto type """
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
