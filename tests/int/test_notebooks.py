import itertools
import json
import os
import subprocess
import sys
from os.path import dirname, realpath
import shutil

import cudf
import pytest

TEST_PATH = dirname(dirname(realpath(__file__)))
DATA_START = "/raid/data"

def test_criteo_notebook(tmpdir):
    input_path = os.path.join(DATA_START, "criteo/crit_int_pq")
    output_path = os.path.join(DATA_START, "criteo/crit_test")
    
    _run_notebook(
        tmpdir,
        os.path.join(dirname(TEST_PATH), "examples", "criteo-example.ipynb"),
        input_path,
        output_path,
        # disable rmm.reinitialize, seems to be causing issues
        transform=lambda line: line.replace("rmm.reinitialize(", "# rmm.reinitialize("),
    )

    
def test_optimize_criteo(tmpdir):
    input_path = os.path.join(DATA_START, "crit_orig")
    output_path = os.path.join(DATA_START, "crit_test_opt")


    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "optimize_criteo.ipynb")
    _run_notebook(
        tmpdir, 
        notebook_path,
        input_path,
        output_path,
    )

    

def test_rossman_example(tmpdir):
    pytest.importorskip("tensorflow")
    input_path = os.path.join(DATA_START, "rossman/input")
    output_path = os.path.join(DATA_START, "rossman/output")
    
    notebookpre_path = os.path.join(
        dirname(TEST_PATH), "examples", "rossmann-store-sales-preprocess.ipynb"
    )
    
    _run_notebook(
        tmpdir, 
        notebookex_path, 
        None,
        input_path,
    )    
    

    notebookex_path = os.path.join(
        dirname(TEST_PATH), "examples", "rossmann-store-sales-example.ipynb"
    )
    _run_notebook(
        tmpdir, 
        notebookex_path, 
        input_path,
        output_path,
    )


def test_gpu_benchmark(tmpdir):
    input_path = os.path.join(DATA_START, "crit_orig")
    output_path = os.path.join(DATA_START, "crit_test_opt")

    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "gpu_benchmark.ipynb", input_path, output_path)
    _run_notebook(
        tmpdir, 
        notebook_path,
        input_path,
        output_path,
    )
    

def _run_notebook(tmpdir, notebook_path, input_path, output_path, transform=None):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    os.environ["INPUT_DATA_DIR"] = input_path
    os.environ["OUTPUT_DATA_DIR"] = output_path
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
    
    # clear out products
    shutil.rmtree(out_put)


def _get_random_criteo_data(rows):
    dtypes = {col: float for col in [f"I{x}" for x in range(1, 14)]}
    dtypes.update({col: int for col in [f"C{x}" for x in range(1, 27)]})
    dtypes["label"] = int
    ret = cudf.datasets.randomdata(rows, dtypes=dtypes)
    # binarize the labels
    ret.label = ret.label % 2
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
