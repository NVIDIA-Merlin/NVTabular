import argparse
import itertools
import json
import os
import subprocess
import sys

parser = argparse.ArgumentParser(
    description="NVTabular TensorFlow Data Loader Benchmark",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--preprocess", type=str, default="no", choices=["no", "yes"], help="Should preprocess dataset"
)
parser.add_argument("--input-dir", type=str, default="", help="Path to input dataset")
parser.add_argument("--output-dir", type=str, default="", help="Path to store the dataset")
parser.add_argument(
    "--dl-type",
    type=str,
    default="NVTabular",
    choices=["TensorFlow", "NVTabular"],
    help="Choose data loader",
)
parser.add_argument(
    "--device", type=str, default="gpu", choices=["cpu", "gpu"], help="Choose device"
)
parser.add_argument(
    "--benchmark-type",
    type=str,
    default="time_training",
    choices=["time_only_dl", "time_training", "convergence_train_loss", "convergence_valid_loss"],
    help="Choose benchmark type",
)
parser.add_argument(
    "--amp", type=str, default="noamp", choices=["noamp", "amp"], help="Choose precision"
)
args = parser.parse_args()
print(args)


def replace_lines(line, replaces):
    for repl in replaces:
        line = line.replace(repl[0], repl[1])
    return line


def _run_notebook(tmpdir, notebook_path, transform=None):
    # read in the notebook as JSON, and extract a python script from it
    notebook = json.load(open(notebook_path))
    source_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    lines = [
        transform(line.rstrip()) if transform else line
        for line in itertools.chain(*source_cells)
        if not (line.startswith("%") or line.startswith("!"))
    ]

    # save the script to a file, and run with the current python executable
    # we're doing this in a subprocess to avoid some issues using 'exec'
    # that were causing a segfault with globals of the exec'ed function going
    # out of scope
    script_path = os.path.join(tmpdir, "notebook.py")
    with open(script_path, "w") as script:
        script.write("\n".join(lines))
    subprocess.check_output([sys.executable, script_path])


if args.input_dir == "":
    raise ValueError("input-dir is not set: " + str(args.input_dir))
if args.output_dir == "":
    raise ValueError("output-dir is not set: " + str(args.output_dir))

replaces = [
    ["INPUT_DIR = '/raid/data/criteo/input/'", "INPUT_DIR = '" + args.input_dir + "'"],
    ["OUTPUT_DIR = '/raid/data/criteo/'", "OUTPUT_DIR = '" + args.output_dir + "'"],
]
if args.preprocess == "yes":
    _run_notebook("/tmp/", "benchmark-preprocess.ipynb", lambda line: replace_lines(line, replaces))

replaces.append(["DL_TYPE = 'NVTabular'", "DL_TYPE = '" + args.dl_type + "'"])
replaces.append(
    ["BENCHMARK_TYPE = 'time_training'", "BENCHMARK_TYPE = '" + args.benchmark_type + "'"]
)
if args.amp == "noamp":
    replaces.append(["AMP = False", "AMP = False"])
if args.amp == "amp":
    replaces.append(["AMP = False", "AMP = True"])
if args.device == "gpu":
    replaces.append(["CPU = False", "CPU = False"])
if args.device == "cpu":
    replaces.append(["CPU = False", "CPU = True"])
print(replaces)
_run_notebook("/tmp/", "benchmark-training.ipynb", lambda line: replace_lines(line, replaces))
