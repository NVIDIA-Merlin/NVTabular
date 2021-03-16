# External dependencies
import os
from os import path

import cudf
from sklearn.model_selection import train_test_split

import nvtabular as nvt

BASE_DIR = "./data/"

if not path.exists(BASE_DIR + "ml-25m"):
    os.makedirs(BASE_DIR, exist_ok=True)
    zip_path = os.path.join(BASE_DIR, "ml-25m.zip")
    if not path.exists(zip_path):
        os.system("mkdir -p " + BASE_DIR)
        os.system("wget http://files.grouplens.org/datasets/movielens/ml-25m.zip")
        os.system("mv ml-25m.zip " + BASE_DIR)

    os.system("unzip " + zip_path + " -d " + BASE_DIR)

movies = cudf.read_csv(os.path.join(BASE_DIR, "ml-25m/movies.csv"))
movies.head()

movies["genres"] = movies["genres"].str.split("|")
movies = movies.drop("title", axis=1)

movies.to_parquet(os.path.join(BASE_DIR, "ml-25m", "movies_converted.parquet"))

ratings = cudf.read_csv(os.path.join(BASE_DIR, "ml-25m", "ratings.csv"))

ratings = ratings.drop("timestamp", axis=1)
train, valid = train_test_split(ratings, test_size=0.2, random_state=42)

train.to_parquet(os.path.join(BASE_DIR, "train.parquet"))
valid.to_parquet(os.path.join(BASE_DIR, "valid.parquet"))

movies = cudf.read_parquet(os.path.join(BASE_DIR, "ml-25m", "movies_converted.parquet"))

joined = ["userId", "movieId"] >> nvt.ops.JoinExternal(movies, on=["movieId"])

cat_features = joined >> nvt.ops.Categorify()

ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))

output = cat_features + ratings

proc = nvt.Workflow(output)

train_iter = nvt.Dataset([os.path.join(BASE_DIR, "train.parquet")], part_size="100MB")
valid_iter = nvt.Dataset([os.path.join(BASE_DIR, "valid.parquet")], part_size="100MB")

proc.fit(train_iter)

proc.transform(train_iter).to_parquet(
    output_path=os.path.join(BASE_DIR, "train"), out_files_per_proc=7
)
proc.transform(valid_iter).to_parquet(
    output_path=os.path.join(BASE_DIR, "valid"), out_files_per_proc=7
)

proc.save(os.path.join(BASE_DIR, "workflow"))
