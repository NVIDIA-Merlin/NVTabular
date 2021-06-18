import os

import abc
from ..column_group import ColumnGroup
from ..io.dataset import DatasetCollection
from ..workflow import Workflow


class TabularDataset:
    def __init__(self, data_dir, client=None, is_dask=False):
        self.data_dir = data_dir
        self.client = client
        self.transformed_dir = os.path.join(data_dir, "transformed")
        self.input_dir = os.path.join(data_dir, "orig")
        self.is_dask = is_dask
        self.transformed = None

        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

    @abc.abstractmethod
    def create_input_column_group(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_default_transformations(self, data) -> ColumnGroup:
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare(self) -> DatasetCollection:
        raise NotImplementedError()

    @property
    def column_group(self) -> ColumnGroup:
        return self.create_input_column_group()

    @property
    def data(self):
        return self.prepare()

    def transform(self, workflow=None, overwrite=False, save=True, to_fit="train") -> DatasetCollection:
        splits = self.prepare()
        if not workflow:
            workflow = Workflow(self.create_default_transformations(splits), self.transformed_dir)

        splits = splits.splits if splits.get("splits") else splits

        splits = workflow.fit_transform_collection(splits, to_fit=to_fit, overwrite=overwrite, save=save)
        self.transformed = splits

        return splits
