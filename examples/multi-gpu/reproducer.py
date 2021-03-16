import cupy
import glob
import os

# gpu_to_use = 0     # Works
gpu_to_use = 1  # Gives cudf/dlpack error

# Need this to avoid device mismatch errors:
cupy.cuda.Device(gpu_to_use).use()

import tensorflow as tf  # noqa: isort

# Horovod's suggested way of handling device selection and memory usage
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_to_use], "GPU")

import nvtabular as nvt  # noqa: isort
from nvtabular.loader.tensorflow import KerasSequenceLoader  # noqa: isort

BASE_DIR = os.path.expanduser("./data/")

BATCH_SIZE = 1024  # Batch Size
CATEGORICAL_COLUMNS = ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = ["genres"]  # Multi-hot
NUMERIC_COLUMNS = []

# Output from ETL-with-NVTabular
TRAIN_PATHS = sorted(glob.glob(os.path.join(BASE_DIR, "train", "*.parquet")))

train_dataset_tf = KerasSequenceLoader(
    TRAIN_PATHS,  # you could also use a glob pattern
    batch_size=BATCH_SIZE,
    label_names=["rating"],
    cat_names=CATEGORICAL_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.06,  # how many batches to load at once
    parts_per_chunk=1,
)

first = train_dataset_tf.__getitem__(0)
