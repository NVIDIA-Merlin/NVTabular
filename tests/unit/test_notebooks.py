import gzip
import itertools
import json
import os
import shutil
from os.path import dirname, realpath


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


def test_rossman_notebook(tmpdir):
    # datasets are stored zipped to avoid blowing up the repo, handle
    test_path = dirname(dirname(realpath(__file__)))
    dataset_path = os.path.join(test_path, "datasets", "rossmann")
    for filename in os.listdir(dataset_path):
        with gzip.open(os.path.join(dataset_path, filename), "rb") as input_file:
            with open(os.path.join(tmpdir, filename[:-3]), "wb") as output_file:
                shutil.copyfileobj(input_file, output_file)

    # read in the notebook
    notebook_path = os.path.join(
        dirname(test_path), "examples", "rossmann-store-sales-example.ipynb"
    )
    source = _read_notebook_source(notebook_path)
    source = source.replace("EPOCHS = 25", "EPOCHS = 1")

    # execute the notebook, passing in the datadir to the toy dataset
    os.environ["DATA_DIR"] = str(tmpdir)
    exec(source, {})
