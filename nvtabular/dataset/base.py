import glob
import os

import abc

from ..workflow import Workflow
import cudf
import dask_cudf


class PublicDataset:
    def __init__(self, data_dir, is_dask=False):
        self.data_dir = data_dir
        self.transformed_dir = os.path.join(data_dir, "transformed")
        self.input_dir = os.path.join(data_dir, "orig")
        self.is_dask = is_dask
        self.transformed = None

    @abc.abstractmethod
    def create_input_column_group(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_default_transformations(self) -> Workflow:
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare(self):
        raise NotImplementedError()

    @property
    def column_group(self):
        return self.create_input_column_group()

    def read_parquet(self, path):
        if os.path.isdir(path):
            path = glob.glob(os.path.join(path, "*.parquet"))

        return dask_cudf.read_parquet(path[0]) if self.is_dask else cudf.read_parquet(path)

    def df(self):
        self.prepare()

        return self.read_parquet(self.data_parquet)

    def transform(self, workflow=None, overwrite=False):
        if not workflow:
            workflow = self.create_default_transformations()
        train_path, eval_path = self.create_splits()

        output = workflow(train_path, eval_path, self.transformed_dir, overwrite=overwrite)
        self.transformed = output

        return output

    def train_df(self, transformed=True):
        if not transformed:
            train_path, _ = self.create_splits()

            return self.read_parquet(train_path)

        if not self.transformed:
            raise ValueError("Dataset not transformed yet")

        return self.read_parquet(self.transformed.train_path)

    def eval_df(self, transformed=True):
        if not transformed:
            _, eval_path = self.create_splits()

            return self.read_parquet(eval_path)

        if not self.transformed:
            raise ValueError("Dataset not transformed yet")

        return self.read_parquet(self.transformed.eval_path)