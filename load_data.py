import os

import cudf
import rmm

from nvtabular.loader.tensorflow import KerasSequenceLoader

INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/end-to-end-poc/data/")
)

rmm.reinitialize(managed_memory=True)

examples = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, "grouped_examples.parquet"))
test_data = examples[["sampled_tag", "movieId", "movieId_count", "target_item"]]
test_data.to_parquet(os.path.join(INPUT_DATA_DIR, "test_data.parquet"))


BATCH_SIZE = 16
CATEGORICAL_COLUMNS = []
CATEGORICAL_MH_COLUMNS = ["sampled_tag", "movieId"]
NUMERIC_COLUMNS = ["movieId_count"]

os.environ["TF_MEMORY_ALLOCATION"] = "0.5"

train_dataset_tf = KerasSequenceLoader(
    os.path.join(INPUT_DATA_DIR, "test_data.parquet"),
    batch_size=BATCH_SIZE,
    label_names=["target_item"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.25,
    parts_per_chunk=1,
)

batch = next(iter(train_dataset_tf))

print(batch)

rmm.reinitialize(managed_memory=False)
