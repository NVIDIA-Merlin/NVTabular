import logging
import os
from glob import glob

import cudf
from kaggle import api as kaggle_api

from nvtabular import ops
from nvtabular.column import Column
from nvtabular.column_group import ColumnGroup, Tag
from nvtabular.dataset.base import ParquetPathCollection, TabularDataset
from nvtabular.dataset.utils import (
    SessionDay,
    create_session_groupby_aggs,
    remove_consecutive_interactions,
)
from nvtabular.io import Dataset

LOG = logging.getLogger("nvtabular")


class YooChoose(TabularDataset):
    ORIG_FILES = ["yoochoose-buys.dat", "yoochoose-clicks.dat", "yoochoose-test.dat"]

    def __init__(
        self,
        work_dir,
        minimum_session_length=2,
        tokenizer=None,
        client_fn=None,
        test_size=0.1,
        random_state=42,
    ):
        super().__init__(os.path.join(work_dir, self.name()), client_fn=client_fn)
        self.minimum_session_length = minimum_session_length
        self.test_size = test_size
        self.random_state = random_state
        self.tokenizer = tokenizer
        self.splits_dir = os.path.join(self.data_dir, "splits")
        if not os.path.exists(self.splits_dir):
            os.makedirs(self.splits_dir)
        self.parquet_dir = os.path.join(self.data_dir, "parquet")
        if not os.path.exists(self.parquet_dir):
            os.makedirs(self.parquet_dir)

    def create_input_column_group(self):
        return ColumnGroup([])

    def create_default_transformations(self, data) -> ColumnGroup:
        features = ColumnGroup(["session_id", "timestamp"])

        # Temporal features
        features += ["timestamp"] >> ops.ItemRecency()
        features += features["timestamp/age_days"] >> ops.LogNormalize(auto_renaming=True)
        features += ["timestamp"] >> ops.TimestampFeatures(add_cycled=True, delimiter="/")

        # Categorical features
        features += [
            Column("item_id", tags=Tag.ITEM_ID),
            Column("category", tags=Tag.ITEM),
        ] >> ops.Categorify()

        # Group-by session
        session_features = features >> ops.Groupby(
            groupby_cols=["session_id"],
            sort_cols=["ts"],
            aggs=create_session_groupby_aggs(
                features,
                extra_aggs=dict(item_id="count", ts=["first", "last"], timestamp="first"),
                to_ignore=["timestamp"],
            ),
            name_sep="/",
        )
        rename_cols = {"item_id/count": "session_size", "timestamp/first": "session_start"}
        session_features = session_features >> ops.Rename(lambda col: rename_cols.get(col, col))
        session_features += session_features["session_start"] >> SessionDay()

        filtered_sessions = session_features >> ops.Filter(
            f=lambda df: df["session_size"] >= self.minimum_session_length
        )

        return filtered_sessions

    def name(self) -> str:
        return "yoochoose"

    @staticmethod
    def _process_clicks(data_path):
        LOG.info(f"Processing {data_path}...")
        df = cudf.read_csv(
            data_path,
            sep=",",
            names=["session_id", "timestamp", "item_id", "category"],
            parse_dates=["timestamp"],
        )
        df = remove_consecutive_interactions(df)
        df = ops.ItemRecency.add_first_seen_col_to_df(df)

        return df

    def prepare(self, frac_size=0.10) -> ParquetPathCollection:
        kaggle_api.authenticate()

        data_files = [f.split("/")[-1] for f in sorted(glob(os.path.join(self.input_dir, "*.dat")))]

        if not all(name in data_files for name in self.ORIG_FILES):
            LOG.info("Downloading data from Kaggle...")
            kaggle_api.dataset_download_files(
                "chadgostopp/recsys-challenge-2015", path=self.input_dir, unzip=True
            )

            df = self._process_clicks(os.path.join(self.input_dir, "yoochoose-clicks.dat"))
            df.to_parquet(os.path.join(self.parquet_dir, "yoochoose-clicks.parquet"))

        train_path, eval_path = self.maybe_create_splits_with_cudf(
            input_dir=os.path.join(self.parquet_dir, "yoochoose-clicks.parquet"),
            output_dir=self.splits_dir,
            test_size=0.1,
        )

        test_path = os.path.join(self.splits_dir, "test")

        if not os.path.exists(test_path):
            LOG.info("Creating test split...")
            df = self._process_clicks(os.path.join(self.input_dir, "yoochoose-test.dat"))
            Dataset(df).to_parquet(test_path)

        return ParquetPathCollection.from_splits(train_path, eval=eval_path, test=test_path)
