import os
import glob
import cudf
import cupy as cp
import nvtabular as nvt
from nvtabular import ColumnGroup
from nvtabular.loader.tensorflow import KerasSequenceLoader, KerasSequenceValidater
from nvtabular.framework_utils.tensorflow import make_feature_column_workflow, layers

import tensorflow as tf
from sklearn.model_selection import train_test_split


dataset_url = "http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
data_dir = os.environ.get(
    "DATA_DIR", "/home/ronayak/ronaya/NVTabular/examples/tensorflow/datasets/petfinder-mini"
)
csv_file = os.path.join(data_dir, "petfinder-mini.csv")

tf.keras.utils.get_file(
    "petfinder_mini.zip",
    dataset_url,
    extract=True,
    cache_dir="/home/ronayak/ronaya/NVTabular/examples/tensorflow/",
)
dataframe = cudf.read_csv(csv_file)
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print(len(train), "train examples")
print(len(val), "validation examples")
print(len(test), "test examples")

# In the original dataset "4" indicates the pet was not adopted.
dataframe["target"] = cp.where(dataframe["AdoptionSpeed"] == 4, 0, 1)

# Drop un-used columns.
dataframe = dataframe.drop(columns=["AdoptionSpeed", "Description"])


def df_to_dataset(df, shuffle, batch_size=32):
    ds = nvt.Dataset(df)
    return KerasSequenceLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        cat_names=[],  # replace these with your categorical feature names,
        cont_names=[],  # replace these with your continuous feature names,
        label_names=["target"],
    )


for df, split in zip([train, val, test], ["train", "valid", "test"]):
    filename = "{}/{}.parquet".format(data_dir, split)
    df.to_parquet(filename)

cat_names = ColumnGroup(
    [
        "Type",
        "Breed1",
        "Color1",
        "Color2",
        "MaturitySize",
        "Gender",
        "FurLength",
        "Vaccinated",
        "Sterilized",
        "Health",
    ]
)
cont_names = ColumnGroup(["Age", "Fee", "PhotoAmt"])
label = ColumnGroup(["AdoptionSpeed"])

# workflow = nvt.Workflow(cat_names=cat_names, cont_names=cont_names, label_name=[])
# workflow.add_cat_preprocess(nvt.ops.LambdaOp(
#     "encode",
#     f=lambda col: cp.where(col == 4, 0, 1),
#     columns=["AdoptionSpeed"],
#     replace=True
# ))
label_feature = label >> (lambda col: cp.where(col == 4, 0, 1))
workflow = nvt.Workflow(cat_names + cont_names + label_feature)

for split in ["train", "valid", "test"]:
    output_path = os.path.join(data_dir, "processed", split)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dataset = nvt.Dataset(os.path.join(data_dir, split + ".parquet"))
    if split == "train":
        shuffle = nvt.io.Shuffle.PER_WORKER
        workflow.fit(dataset)
        workflow.transform(dataset).to_parquet(
            output_path=output_path, shuffle=shuffle, out_files_per_proc=1
        )
    else:
        shuffle = False
        workflow.transform(dataset).to_parquet(
            output_path=output_path, shuffle=None, out_files_per_proc=1
        )

animal_type = tf.feature_column.categorical_column_with_vocabulary_list("Type", ["Cat", "Dog"])

feature_columns = []

# numeric cols
for header in ["PhotoAmt", "Fee", "Age"]:
    feature_columns.append(tf.feature_column.numeric_column(header))


age = tf.feature_column.numeric_column("Age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[1, 2, 3, 4, 5])
feature_columns.append(age_buckets)

indicator_column_names = [
    "Type",
    "Color1",
    "Color2",
    "Gender",
    "MaturitySize",
    "FurLength",
    "Vaccinated",
    "Sterilized",
    "Health",
]
for col_name in indicator_column_names:
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        col_name, dataframe[col_name].unique().to_pandas()
    )
    indicator_column = tf.feature_column.indicator_column(categorical_column)
    feature_columns.append(indicator_column)

breed1 = tf.feature_column.categorical_column_with_vocabulary_list(
    "Breed1", dataframe.Breed1.unique().to_pandas()
)
breed1_embedding = tf.feature_column.embedding_column(breed1, dimension=8)
feature_columns.append(breed1_embedding)

age_type_feature = tf.feature_column.crossed_column(
    [age_buckets, animal_type], hash_bucket_size=100
)
feature_columns.append(tf.feature_column.indicator_column(age_type_feature))

online_workflow, feature_columns = make_feature_column_workflow(feature_columns, "AdoptionSpeed")
feature_layer = layers.DenseFeatures(feature_columns)


def make_nvt_dataset(split, shuffle=True, batch_size=32):
    train_paths = glob.glob(os.path.join(data_dir, "processed", split, "*.parquet"))
    dataset = nvt.Dataset(train_paths, engine="parquet")
    if split == "train":
        online_workflow.fit(dataset)
    ds = KerasSequenceLoader(
        online_workflow.transform(dataset),
        batch_size=batch_size,
        feature_columns=feature_columns,
        label_names=["AdoptionSpeed"],
        shuffle=shuffle,
        buffer_size=0.02,
        parts_per_chunk=1,
    )
    return ds


train_ds = make_nvt_dataset("train", shuffle=True)
val_ds = make_nvt_dataset("valid", shuffle=False, batch_size=2048)
test_ds = make_nvt_dataset("test", shuffle=False, batch_size=4096)

train_ds.data.to_ddf().columns

inputs = {}
for column in feature_columns:
    column = getattr(column, "categorical_column", column)
    dtype = getattr(column, "dtype", tf.int64)
    inputs[column.key] = tf.keras.Input(name=column.key, shape=(1,), dtype=dtype)

x = feature_layer(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(1)(x)

print(inputs.values())
model = tf.keras.Model(inputs=inputs.values(), outputs=x)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_ds, callbacks=[KerasSequenceValidater(val_ds)], epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)
