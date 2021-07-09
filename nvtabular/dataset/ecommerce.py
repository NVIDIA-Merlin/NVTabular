import os

import cudf
from kaggle import api as kaggle_api
from sklearn.model_selection import train_test_split

from nvtabular import ops
from nvtabular.column_group import ColumnGroup, Tag
from nvtabular.dataset.base import ParquetPathCollection, TabularDataset
from nvtabular.io import Dataset


class ClothingReviews(TabularDataset):
    ORIG_FILE_NAME = "Womens Clothing E-Commerce Reviews.csv"
    PARQUET_FILE_NAME = "Womens Clothing E-Commerce Reviews.parquet"

    def __init__(self, work_dir, tokenizer=None, client_fn=None, test_size=0.1, random_state=42):
        super().__init__(os.path.join(work_dir, self.name()), client_fn=client_fn)
        self.parquet_dir = os.path.join(self.input_dir, "parquet")
        self.data_parquet = os.path.join(self.parquet_dir, self.PARQUET_FILE_NAME)
        self.data_csv = os.path.join(self.input_dir, self.ORIG_FILE_NAME)

        self.test_size = test_size
        self.random_state = random_state
        self.tokenizer = tokenizer
        self.splits_dir = os.path.join(self.data_dir, "splits")
        if not os.path.exists(self.splits_dir):
            os.makedirs(self.splits_dir)

    def create_input_column_group(self):
        columns = ColumnGroup([])
        columns += ColumnGroup(["Title", "Review Text"], tags=Tag.TEXT)
        columns += ColumnGroup(
            ["Division Name", "Department Name", "Class Name", "Clothing ID"], tags=Tag.CATEGORICAL
        )
        columns += ColumnGroup(["Positive Feedback Count", "Age"], tags=Tag.CONTINUOUS)

        columns += (
            ColumnGroup(["Recommended IND"])
            >> ops.Rename(f=lambda x: x.replace(" IND", ""))
            >> ops.AddMetadata(tags=Tag.TARGETS_BINARY)
        )
        columns += ColumnGroup(["Rating"], tags=Tag.TARGETS_REGRESSION)

        return columns

    def create_default_transformations(self, data) -> ColumnGroup:
        outputs = self.column_group.targets_column_group
        outputs += self.column_group.continuous_column_group >> ops.FillMissing() >> ops.Normalize()
        outputs += self.column_group.categorical_column_group >> ops.Categorify()

        if self.tokenizer:
            if isinstance(self.tokenizer, ops.TokenizeText):
                outputs += self.column_group.text_column_group >> self.tokenizer
            else:
                outputs += self.column_group.text_column_group >> ops.TokenizeText(
                    self.tokenizer,
                    max_length=200,
                    do_lower=False,
                    cache_dir=os.path.join(self.data_dir, "tokenizers"),
                    do_truncate=True,
                )
        else:
            outputs += self.column_group.text_column_group

        return outputs

    def name(self) -> str:
        return "clothing_reviews"

    def prepare(self, frac_size=0.10) -> ParquetPathCollection:
        kaggle_api.authenticate()

        if not os.path.exists(self.data_parquet):
            if not os.path.exists(self.data_csv):
                kaggle_api.dataset_download_files(
                    "nicapotato/womens-ecommerce-clothing-reviews", path=self.input_dir, unzip=True
                )

            dataset = Dataset(
                self.data_csv,
                engine="csv",
                part_mem_fraction=frac_size,
                client=self.client,
            )
            dataset.to_parquet(self.parquet_dir, preserve_files=True)

        train_path = os.path.join(self.splits_dir, "train")
        eval_path = os.path.join(self.splits_dir, "eval")

        if not os.path.exists(train_path) or not os.path.exists(eval_path):
            df = cudf.read_parquet(self.data_parquet)
            train, eval = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )
            Dataset(train).to_parquet(train_path)
            Dataset(eval).to_parquet(eval_path)

        return ParquetPathCollection.from_splits(train_path, eval=eval_path)
