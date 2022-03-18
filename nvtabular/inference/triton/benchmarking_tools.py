""" utilities for running the triton perf_analyzer on a nvtabulat model """
import json
import os
import subprocess
import tempfile

import pandas as pd

import nvtabular as nvt
from merlin.core.dispatch import is_string_dtype


def run_perf_analyzer(model_path, input_data_path, num_rows=10, model_version=1):
    """Runs perf_analyzer and returns a dataframe with statistics from it

    Parameters
    ----------
    model_path : str
        The fullpath to the model to analyze.
    input_data_path: str
        Path to datafiles containing example data to query the model with. Can
        be anything we can pass to a nvt.Dataset object (csv file/parquet etc)
    num_rows: int
        How many rows to query for
    model_version: int
        Which model version to use
    """
    # load the workflow and get the base perf analyzer commandline
    model_name = os.path.basename(model_path)

    workflow_path = os.path.join(model_path, str(model_version), "workflow")
    workflow = nvt.Workflow.load(workflow_path)
    cmdline = _get_perf_analyzer_commandline(workflow, model_name, batch_size=num_rows)

    # read in the input data and write out as a JSON file
    df = nvt.Dataset(input_data_path).to_ddf().head(num_rows)
    json_data = _convert_df_to_triton_json(df, workflow.input_dtypes)

    with tempfile.NamedTemporaryFile("w", suffix=".json") as json_file:
        json.dump(json_data, json_file, indent=2)
        cmdline.extend(["--input-data", json_file.name])
        json_file.flush()

        with tempfile.NamedTemporaryFile("w", suffix=".csv") as csv_report:
            csv_report.close()
            cmdline.extend(["-f", csv_report.name])
            result = subprocess.run(cmdline, stdout=subprocess.PIPE, check=True, encoding="utf8")
            print(result.stdout)
            return pd.read_csv(csv_report.name)


def _convert_df_to_triton_json(df, input_dtypes=None):
    """perf_analzyer requires input data (when we have strings+integers at least). This function
    converts a cudf dataframe to a jsonable representation that perf_analyzer can use"""
    input_dtypes = input_dtypes or df.dtypes

    json_columns = {}
    for col in input_dtypes:
        values = df[col].values_host
        # we need to fill None values in strings
        if is_string_dtype(values.dtype):
            json_columns[col] = [v if v is not None else "" for v in values]
        else:
            json_columns[col] = values.tolist()
    return {"data": [json_columns]}


def _get_perf_analyzer_commandline(workflow, modelname, batch_size=10):
    """perf_analyzer requires us to specify the shape for every single input column, which is
    difficult for models (aside from movielens). this returns the commandline needed to run
    perf analyzer"""
    cmdline = ["perf_analyzer", "-m", modelname, "-i", "grpc"]
    for col in workflow.input_dtypes:
        cmdline.extend(
            [
                "\\\n\t",
                "--shape",
                f"{col}:{batch_size},1",
            ]
        )

    return cmdline


def generate_graph(model_times, metric="p95 latency"):
    """utility function to generate graphs of times. model_times is a dictionary of
    label to pandas dataframe"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for model_name, perf_report in model_times.items():
        batch_size = perf_report["batch_size"].values
        latency_ms = perf_report[metric].values / 1000
        ax.plot(batch_size, latency_ms, label=model_name, marker="o", markersize=3)

    ax.set_ylim(0)
    ax.set_ylabel(f"{metric} (ms)")
    ax.set_xlabel("batch size")
    ax.set_xscale("log")
    ax.legend()

    fig.set_dpi(100)
    return fig, ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate perf statistics on a nvtabular triton model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        dest="input_path",
        required=True,
        help="input datafile (csv/parquet etc) used to generate requests",
    )
    parser.add_argument("--model_name", type=str, default="rossmann_nvt", dest="model_name")
    parser.add_argument("--model_path", type=str, default="/models", dest="model_path")
    parser.add_argument("--output", type=str, default="report.csv", dest="output")
    args = parser.parse_args()

    batch_sizes = [2 ** x for x in range(3, 16)]
    dfs = []
    for batch_size in batch_sizes:
        data = run_perf_analyzer(
            os.path.join(args.model_path, args.model_name), args.input_path, num_rows=batch_size
        )
        data["batch_size"] = [batch_size]
        print(data)
        dfs.append(data)

    report = pd.concat(dfs)
    report.reset_index()
    report.to_csv(args.output)
