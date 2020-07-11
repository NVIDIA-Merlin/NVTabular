import itertools
import json
import os
from os.path import dirname, realpath

import cudf

TEST_PATH = dirname(dirname(realpath(__file__)))


def test_criteo_notebook(tmpdir):
    for i in range(24):
        df = _get_random_criteo_data(10000)
        df.to_parquet(os.path.join(tmpdir, f"day_{i}.parquet"))

    # read in the notebook
    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "criteo-example.ipynb")
    os.environ["INPUT_DATA_DIR"] = str(tmpdir)
    os.environ["OUTPUT_DATA_DIR"] = str(tmpdir)
    source = _read_notebook_source(notebook_path)
    exec(source, {})  # pylint: disable=exec-used


def test_optimize_criteo(tmpdir):
    _get_random_criteo_data(1000).to_csv(os.path.join(tmpdir, "day_0"), sep="\t", header=False)

    # read in the notebook
    notebook_path = os.path.join(dirname(TEST_PATH), "examples", "optimize_criteo.ipynb")
    os.environ["INPUT_PATH"] = str(tmpdir)
    os.environ["OUTPUT_PATH"] = str(tmpdir)
    source = _read_notebook_source(notebook_path)
    exec(source, {})  # pylint: disable=exec-used


def test_rossman_example(tmpdir):
    _get_random_rossmann_data(1000).to_csv(os.path.join(tmpdir, "train.csv"))
    _get_random_rossmann_data(1000).to_csv(os.path.join(tmpdir, "valid.csv"))

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


def _get_random_criteo_data(rows):
    dtypes = {col: float for col in [f"I{x}" for x in range(1, 14)]}
    dtypes.update({col: int for col in [f"C{x}" for x in range(1, 27)]})
    dtypes["label"] = int
    return cudf.datasets.randomdata(rows, dtypes=dtypes)


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
