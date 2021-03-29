#
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

import itertools
import json
import os
import subprocess
import sys
from os.path import dirname, realpath

import cudf
import pytest

from tests.conftest import get_cuda_cluster

TEST_PATH = dirname(dirname(realpath(__file__)))


def test_criteo_notebook(tmpdir):
    tor = pytest.importorskip("fastai")  # noqa
    # create a toy dataset in tmpdir, and point environment variables so the notebook
    # will read from it
    for i in range(24):
        df = _get_random_criteo_data(1000)
        df.to_parquet(os.path.join(tmpdir, f"day_{i}.parquet"))
    os.environ["INPUT_DATA_DIR"] = str(tmpdir)
    os.environ["OUTPUT_DATA_DIR"] = str(tmpdir)

    _run_notebook(
        tmpdir,
        os.path.join(
            dirname(TEST_PATH),
            "examples/scaling-criteo/",
            "02-03b-ETL-with-NVTabular-Training-with-PyTorch.ipynb",
        ),
        # disable rmm.reinitialize, seems to be causing issues
        transform=lambda line: line.replace("rmm.reinitialize(", "# rmm.reinitialize("),
    )


def test_optimize_criteo(tmpdir):
    _get_random_criteo_data(1000).to_csv(os.path.join(tmpdir, "day_0"), sep="\t", header=False)
    os.environ["INPUT_DATA_DIR"] = str(tmpdir)
    os.environ["OUTPUT_DATA_DIR"] = str(tmpdir)

    notebook_path = os.path.join(
        dirname(TEST_PATH),
        "examples/scaling-criteo/",
        "01-Download-Convert.ipynb",
    )
    _run_notebook(tmpdir, notebook_path)


@pytest.mark.skip(reason="Need to install pydot / use mock data on this")
def test_movielens_example(tmpdir):
    os.environ["OUTPUT_DATA_DIR"] = str(tmpdir)
    notebooks = [
        "01-Download-Convert.ipynb",
        "02-ETL-with-NVTabular.ipynb",
        "03a-Training-with-TF.ipynb",
        "03b-Training-with-PyTorch.ipynb",
    ]
    for notebook in notebooks:
        notebook_path = os.path.join(
            dirname(TEST_PATH),
            "examples/getting-started-movielens/",
            notebook,
        )
        _run_notebook(
            tmpdir,
            notebook_path,
            lambda line: line.replace(
                "BASE_DIR = '/raid/data/ml/'", "BASE_DIR = '" + str(tmpdir) + "/'"
            ),
        )


def test_rossman_example(tmpdir):
    _get_random_rossmann_data(1000).to_csv(os.path.join(tmpdir, "train.csv"))
    _get_random_rossmann_data(1000).to_csv(os.path.join(tmpdir, "valid.csv"))
    os.environ["OUTPUT_DATA_DIR"] = str(tmpdir)

    notebook_path = os.path.join(
        dirname(TEST_PATH),
        "examples/tabular-data-rossmann/",
        "02-ETL-with-NVTabular.ipynb",
    )
    _run_notebook(tmpdir, notebook_path)

    os.environ["INPUT_DATA_DIR"] = str(tmpdir)

    notebooks = []
    try:
        import torch  # noqa

        notebooks.append("03b-Training-with-PyTorch.ipynb")
        import fastai  # noqa

        notebooks.append("04-Training-with-FastAI.ipynb")
    except Exception:
        pass
    try:
        import nvtabular.loader.tensorflow  # noqa

        notebooks.append("03a-Training-with-TF.ipynb")
    except Exception:
        pass

    for notebook in notebooks:
        notebook_path = os.path.join(
            dirname(TEST_PATH),
            "examples/tabular-data-rossmann/",
            notebook,
        )
        _run_notebook(tmpdir, notebook_path, lambda line: line.replace("EPOCHS = 25", "EPOCHS = 1"))


def test_multigpu_dask_example(tmpdir):
    with get_cuda_cluster() as cuda_cluster:
        os.environ["BASE_DIR"] = str(tmpdir)
        scheduler_port = cuda_cluster.scheduler_address

        def _nb_modify(line):
            # Use cuda_cluster "fixture" port rather than allowing notebook
            # to deploy a LocalCUDACluster within the subprocess
            line = line.replace("cluster = None", f"cluster = '{scheduler_port}'")
            # Use a much smaller "toy" dataset
            line = line.replace("write_count = 25", "write_count = 4")
            line = line.replace('freq = "1s"', 'freq = "1h"')
            # Use smaller partitions for smaller dataset
            line = line.replace("part_mem_fraction=0.1", "part_size=1_000_000")
            line = line.replace("out_files_per_proc=8", "out_files_per_proc=1")
            return line

        notebook_path = os.path.join(
            dirname(TEST_PATH), "examples/multi-gpu-toy-example/", "multi-gpu_dask.ipynb"
        )
        _run_notebook(tmpdir, notebook_path, _nb_modify)


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


def _get_random_criteo_data(rows):
    dtypes = {col: float for col in [f"I{x}" for x in range(1, 14)]}
    dtypes.update({col: int for col in [f"C{x}" for x in range(1, 27)]})
    dtypes["label"] = bool
    ret = cudf.datasets.randomdata(rows, dtypes=dtypes)
    # binarize the labels
    ret.label = ret.label.astype(int)
    return ret


def _get_random_rossmann_data(rows):
    dtypes = {
        col: int
        for col in [
            "Store",
            "DayOfWeek",
            "Sales",
            "Customers",
            "Open",
            "Promo",
            "SchoolHoliday",
            "Year",
            "Month",
            "Week",
            "Day",
            "CompetitionOpenSinceMonth",
            "CompetitionOpenSinceYear",
            "Promo2",
            "Promo2SinceWeek",
            "Promo2SinceYear",
            "trend",
            "trend_DE",
            "Month_DE",
            "Day_DE",
            "Max_TemperatureC",
            "Mean_TemperatureC",
            "Min_TemperatureC",
            "Dew_PointC",
            "MeanDew_PointC",
            "Min_DewpointC",
            "Max_Humidity",
            "Mean_Humidity",
            "Min_Humidity",
            "Max_Sea_Level_PressurehPa",
            "Mean_Sea_Level_PressurehPa",
            "Min_Sea_Level_PressurehPa",
            "Max_Wind_SpeedKm_h",
            "Mean_Wind_SpeedKm_h",
            "WindDirDegrees",
            "CompetitionDaysOpen",
            "CompetitionMonthsOpen",
            "Promo2Days",
            "Promo2Weeks",
            "State_DE",
            "Id",
        ]
    }  # noqa
    dtypes.update(
        {
            col: object
            for col in [
                "Date",
                "StoreType",
                "Assortment",
                "PromoInterval",
                "State",
                "file",
                "week",
                "file_DE",
                "week_DE",
                "Date_DE",
                "Events",
                "StateName",
                "CompetitionOpenSince",
                "Promo2Since",
            ]
        }
    )  # noqa
    dtypes.update(
        {
            col: float
            for col in [
                "CompetitionDistance",
                "Max_VisibilityKm",
                "Mean_VisibilityKm",
                "Min_VisibilitykM",
                "Max_Gust_SpeedKm_h",
                "Precipitationmm",
                "CloudCover",
                "AfterSchoolHoliday",
                "BeforeSchoolHoliday",
                "AfterStateHoliday",
                "BeforeStateHoliday",
                "AfterPromo",
                "BeforePromo",
                "SchoolHoliday_bw",
                "StateHoliday_bw",
                "Promo_bw",
                "SchoolHoliday_fw",
                "StateHoliday_fw",
                "Promo_fw",
            ]
        }
    )  # noqa
    dtypes["StateHoliday"] = bool
    return cudf.datasets.randomdata(rows, dtypes=dtypes)
