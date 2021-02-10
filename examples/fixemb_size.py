import os
import shutil

import cudf

import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes

df = cudf.DataFrame(
    {
        "author": [
            "User_A",
            "User_A",
            "User_B",
            "User_C",
            "User_A",
            "User_B",
            "User_B",
            "User_C",
            "User_B",
            "User_C",
            "User_K",
            "User_M",
            "User_N",
            "User_m",
            "User_n",
        ],
        "engaging_user": [
            "User_B",
            "User_B",
            "User_A",
            "User_D",
            "User_B",
            "User_k",
            "User_A",
            "User_D",
            "User_D",
            "User_D",
            "User_K",
            "User_M",
            "User_N",
            "User_m",
            "User_n",
        ],
    }
)

df_test = cudf.DataFrame(
    {
        "author": [
            "User_K",
            "User_M",
            "User_M",
            "User_d",
            "User_K",
            "User_K",
            "User_B",
            "User_C",
            "User_B",
            "User_C",
            "User_m",
        ],
        "engaging_user": [
            "User_B",
            "User_b",
            "User_A",
            "User_D",
            "User_B",
            "User_c",
            "User_A",
            "User_D",
            "User_D",
            "User_b",
            "User_n",
        ],
    }
)


PREPROCESS_DIR = os.path.join("./examples/data")
PREPROCESS_DIR_TRAIN = os.path.join(PREPROCESS_DIR, "train")
PREPROCESS_DIR_VALID = os.path.join(PREPROCESS_DIR, "valid")

isDirectory = os.path.isdir(PREPROCESS_DIR_TRAIN)
if isDirectory:
    shutil.rmtree(PREPROCESS_DIR_TRAIN)

cat_names = ["author", "engaging_user"]
CONTINUOUS_COLUMNS = ["price"]
LABEL_COLUMNS = ["platform"]

cat_features = cat_names >> nvt.ops.Categorify(
    max_size={"author": 12, "engaging_user": 8}, num_buckets=5
)
workflow = nvt.Workflow(cat_features)

train_dataset = nvt.Dataset(df)
valid_dataset = nvt.Dataset(df_test)
workflow.fit(train_dataset)
workflow.transform(train_dataset).to_parquet(PREPROCESS_DIR_TRAIN, shuffle=False)
expected = cudf.read_parquet(PREPROCESS_DIR_TRAIN + "/*.parquet")
print(expected)

# transformed_valid = workflow.transform(valid_dataset).to_ddf().compute()
# print(transformed_valid)

embeddings = get_embedding_sizes(workflow)
print(embeddings)
