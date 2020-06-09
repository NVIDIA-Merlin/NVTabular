import os

import cudf

import nvtabular as nvt

DATA_DIR = os.environ.get("DATA_DIR", "/home/ronayak/ronaya/recsys/nvtabular/examples/data/")

CATEGORICAL_COLUMNS = [
    "Store",
    "DayOfWeek",
    "Year",
    "Month",
    "Day",
    "StateHoliday",
    "CompetitionMonthsOpen",
    "Promo2Weeks",
    "StoreType",
    "Assortment",
    "PromoInterval",
    "CompetitionOpenSinceYear",
    "Promo2SinceYear",
    "State",
    "Week",
    "Events",
    "Promo_fw",
    "Promo_bw",
    "StateHoliday_fw",
    "StateHoliday_bw",
    "SchoolHoliday_fw",
    "SchoolHoliday_bw",
]

CONTINUOUS_COLUMNS = [
    "CompetitionDistance",
    "Max_TemperatureC",
    "Mean_TemperatureC",
    "Min_TemperatureC",
    "Max_Humidity",
    "Mean_Humidity",
    "Min_Humidity",
    "Max_Wind_SpeedKm_h",
    "Mean_Wind_SpeedKm_h",
    "CloudCover",
    "trend",
    "trend_DE",
    "AfterStateHoliday",
    "BeforeStateHoliday",
    "Promo",
    "SchoolHoliday",
]
LABEL_COLUMNS = ["Sales"]

COLUMNS = CATEGORICAL_COLUMNS + CONTINUOUS_COLUMNS + LABEL_COLUMNS

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VALID_PATH = os.path.join(DATA_DIR, "valid.csv")

# train_df = pd.read_csv('./data/train.csv')
# test_df = pd.read_csv('./data/test.csv')
# note that here, we want to perform a normalization transformation on the label

# column. Since NVT doesn't support transforming label columns right now, we'll
# pretend it's a regular continuous column during our feature engineering phase
proc = nvt.Workflow(
    cat_names=CATEGORICAL_COLUMNS,
    cont_names=CONTINUOUS_COLUMNS,  # +LABEL_COLUMNS,
    label_name=LABEL_COLUMNS,
)


proc.add_feature(nvt.ops.Dropna())
proc.add_cont_preprocess(nvt.ops.LogOp(columns=["Sales"]))
proc.add_cont_preprocess(nvt.ops.Normalize())
proc.add_cat_preprocess(nvt.ops.Categorify())
proc.finalize()

GPU_MEMORY_FRAC = 0.2
train_ds_iterator = nvt.dataset(TRAIN_PATH, gpu_memory_frac=GPU_MEMORY_FRAC, columns=COLUMNS)
# valid_ds_iterator = nvt.dataset(VALID_PATH, gpu_memory_frac=GPU_MEMORY_FRAC, columns=COLUMNS)

PREPROCESS_DIR = os.path.join(DATA_DIR, "jp_ross")
PREPROCESS_DIR_TRAIN = os.path.join(PREPROCESS_DIR, "train")
PREPROCESS_DIR_VALID = os.path.join(PREPROCESS_DIR, "valid")

# ! mkdir -p $PREPROCESS_DIR_TRAIN
# ! mkdir -p $PREPROCESS_DIR_VALID

proc.apply(
    train_ds_iterator,
    apply_offline=True,
    record_stats=True,
    output_path=PREPROCESS_DIR_TRAIN,
    shuffle=True,
    num_out_files=1,
)

gdf = cudf.read_parquet(
    "/home/ronayak/ronaya/recsys/nvtabular/examples/data/jp_ross/train/0.parquet"
)
print(gdf.shape)
print(gdf.columns)
