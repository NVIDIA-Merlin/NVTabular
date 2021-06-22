import os

import cudf
from sklearn.model_selection import train_test_split

from nvtabular import ops
from nvtabular.column_group import ColumnGroup, Tag
from nvtabular.dataset.base import TabularDataset
from nvtabular.io import Dataset
from nvtabular.io.dataset import DatasetCollection
from nvtabular.tag import TagAs
from nvtabular.utils import download_file


class MovieLens(TabularDataset):
    def __init__(self, work_dir, tokenizer=None, client=None, test_size=0.1, random_state=42):
        super().__init__(os.path.join(work_dir, self.name()))
        self.csv_dir = os.path.join(self.data_dir, "ml-25m")
        self.client = client

        self.test_size = test_size
        self.random_state = random_state
        self.tokenizer = tokenizer
        self.splits_dir = os.path.join(self.data_dir, "splits")
        if not os.path.exists(self.splits_dir):
            os.makedirs(self.splits_dir)

    def create_input_column_group(self):
        columns = ColumnGroup([])
        columns += ColumnGroup(["Title", "Review Text"], tags=Tag.TEXT)
        columns += ColumnGroup(["Division Name", "Department Name", "Class Name", "Clothing ID"],
                               tags=Tag.CATEGORICAL)
        columns += ColumnGroup(["Positive Feedback Count", "Age"], tags=Tag.CONTINUOUS)

        columns += (ColumnGroup(["Recommended IND"])
                    >> ops.Rename(f=lambda x: x.replace(" IND", ""))
                    >> TagAs(Tag.TARGETS_BINARY)
                    )
        columns += ColumnGroup(["Rating"], tags=Tag.TARGETS_REGRESSION)

        return columns

    def create_default_transformations(self, data) -> ColumnGroup:
        user_id = ColumnGroup(["userId"], tags=Tag.USER)
        item_id = ColumnGroup(["movieId"], tags=Tag.ITEM)

        cat_features = (user_id + item_id
                        >> ops.JoinExternal(data.movies, on=["movieId"])
                        >> ops.Categorify()
                        )

        # Make rating a binary target
        rating_binary = (ColumnGroup(["rating"])
                         >> (lambda col: (col > 3).astype("int8"))
                         >> ops.Rename(postfix="_binary")
                         >> TagAs(is_binary_target=True)
                         )

        rating = ColumnGroup(["rating"]) >> TagAs(is_regression_target=True)

        return cat_features + rating_binary + rating

    def name(self) -> str:
        return "movielens"

    def prepare(self, frac_size=0.10) -> DatasetCollection:
        if not os.path.exists(self.csv_dir):
            zip_path = os.path.join(self.data_dir, "ml-25m.zip")
            download_file("http://files.grouplens.org/datasets/movielens/ml-25m.zip", zip_path)

        train_path = os.path.join(self.splits_dir, "train")
        eval_path = os.path.join(self.splits_dir, "eval")
        movies_path = os.path.join(self.splits_dir, "movies")

        if not os.path.exists(train_path) or not os.path.exists(eval_path):
            ratings = cudf.read_csv(os.path.join(self.csv_dir, "ratings.csv"))
            ratings = ratings.drop("timestamp", axis=1)
            train, valid = train_test_split(ratings, test_size=0.2, random_state=42)
            Dataset(train).to_parquet(train_path)
            Dataset(valid).to_parquet(eval_path)

        if not os.path.exists(movies_path):
            movies = cudf.read_csv(os.path.join(self.csv_dir, "movies.csv"))
            movies["genres"] = movies["genres"].str.split("|")
            movies = movies.drop("title", axis=1)
            Dataset(movies).to_parquet(movies_path)

        return DatasetCollection(
            splits=DatasetCollection(
                train=Dataset.from_pattern(train_path),
                eval=Dataset.from_pattern(eval_path)
            ),
            movies=Dataset.from_pattern(movies_path)
        )
