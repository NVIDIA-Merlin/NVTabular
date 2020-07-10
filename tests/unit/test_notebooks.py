import gzip
import itertools
import json
import os
import shutil
from os.path import dirname, realpath

TEST_PATH = dirname(dirname(realpath(__file__)))


def test_optimize_criteo(tmpdir):
    # datasets are stored zipped to avoid blowing up the repo, handle
    dataset_path = os.path.join(TEST_PATH, "datasets", "criteo")
    _unzip_dataset(dataset_path, tmpdir)

    # read in the notebook
    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "optimize_criteo.ipynb")
    os.environ["INPUT_PATH"] = str(tmpdir)
    os.environ["OUTPUT_PATH"] = str(tmpdir)
    source = _read_notebook_source(notebook_path)
    exec(source, {})  # pylint: disable=exec-used


def test_rossman_notebook(tmpdir):
    # datasets are stored zipped to avoid blowing up the repo, handle
    dataset_path = os.path.join(TEST_PATH, "datasets", "rossmann")
    _unzip_dataset(dataset_path, tmpdir)

    # read in the notebook
    notebook_path = os.path.join(
        dirname(TEST_PATH), "examples", "rossmann-store-sales-example.ipynb"
    )
    source = _read_notebook_source(notebook_path)
    source = source.replace("EPOCHS = 25", "EPOCHS = 1")

    # execute the notebook, passing in the datadir to the toy dataset
    os.environ["DATA_DIR"] = str(tmpdir)
    exec(source, {})  # pylint: disable=exec-used


def _read_notebook_source(notebook_path):
    """ reads in the python code from a jupyter notebook """
    notebook = json.load(open(notebook_path))
    source_cells = [cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"]
    lines = [
        line.rstrip()
        for line in itertools.chain(*source_cells)
        if not (line.startswith("%") or line.startswith("!"))
    ]
    return "\n".join(lines)


def _unzip_dataset(dataset_path, tmpdir):
    for filename in os.listdir(dataset_path):
        with gzip.open(os.path.join(dataset_path, filename), "rb") as input_file:
            with open(os.path.join(tmpdir, filename[:-3]), "wb") as output_file:
                shutil.copyfileobj(input_file, output_file)


if __name__ == "__main__":
    test_optimize_criteo("./tmp")
