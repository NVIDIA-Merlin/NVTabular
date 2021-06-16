import os

from kaggle import api as kaggle_api
from sklearn.model_selection import train_test_split

from nvtabular import ops
from nvtabular.column_group import ColumnGroup, Tag
from nvtabular.dataset.base import PublicDataset
from nvtabular.io import Dataset
from nvtabular.workflow import Workflow


class ClothingReviews(PublicDataset):
    ORIG_FILE_NAME = "Womens Clothing E-Commerce Reviews.csv"
    PARQUET_FILE_NAME = "Womens Clothing E-Commerce Reviews.parquet"

    def __init__(self, data_dir, client=None, tokenizer=None, test_size=0.1, random_state=42):
        super().__init__(data_dir)
        self.client = client
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
        columns += ColumnGroup(["Division Name", "Department Name", "Class Name", "Clothing ID"],
                               tags=Tag.CATEGORICAL)
        columns += ColumnGroup(["Positive Feedback Count", "Age"], tags=Tag.CONTINUOUS)

        columns += ColumnGroup(["Recommended IND"], tags=Tag.TARGETS_BINARY)
        columns += ColumnGroup(["Rating"], tags=Tag.TARGETS_REGRESSION)

        return columns

    def create_default_transformations(self) -> Workflow:
        outputs = self.column_group.get_tagged(Tag.TARGETS)
        outputs += self.column_group.get_tagged(Tag.CONTINUOUS) >> ops.FillMissing() >> ops.Normalize()
        outputs += self.column_group.get_tagged(Tag.CATEGORICAL) >> ops.Categorify()

        if self.tokenizer:
            if isinstance(self.tokenizer, ops.TokenizeText):
                outputs += self.column_group.get_tagged(Tag.TEXT) >> self.tokenizer
            else:
                outputs += self.column_group.get_tagged(Tag.TEXT) >> ops.TokenizeText(
                    self.tokenizer,
                    max_length=200,
                    do_lower=False,
                    cache_dir=os.path.join(self.data_dir, "tokenizers"),
                    do_truncate=True)
        else:
            outputs += self.column_group.get_tagged(Tag.TEXT)

        return Workflow(outputs)

    def prepare(self, frac_size=0.10):
        kaggle_api.authenticate()

        if not os.path.exists(self.data_parquet):
            if not os.path.exists(self.data_csv):
                kaggle_api.dataset_download_files("nicapotato/womens-ecommerce-clothing-reviews", path=self.input_dir,
                                                  unzip=True)

            dataset = Dataset(
                self.data_csv,
                engine="csv",
                part_mem_fraction=frac_size,
                client=self.client,
            )
            dataset.to_parquet(self.parquet_dir, preserve_files=True)

    def create_splits(self):
        train_path = os.path.join(self.splits_dir, "train.parquet")
        eval_path = os.path.join(self.splits_dir, "eval.parquet")

        if not os.path.exists(train_path):
            train, valid = train_test_split(self.df(), test_size=self.test_size, random_state=self.random_state)
            train.to_parquet(train_path)
            valid.to_parquet(eval_path)

        return train_path, eval_path
