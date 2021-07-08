import abc
import contextlib
import logging
import os
import warnings
from glob import glob
from types import SimpleNamespace
from typing import Any, Optional, Union

import joblib
import rmm
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from nvtabular.io import Dataset
from nvtabular.utils import Namespace, _pynvml_mem_size, device_mem_size

from ..column_group import ColumnGroup
from ..io.dataset import DatasetCollection
from ..ops import Schema
from ..ops.statistics import DatasetCollectionStatistics, Statistics
from ..workflow import Workflow

LOG = logging.getLogger("nvtabular")


class ParquetPathCollection(Namespace):
    def load(self, transformed=True, **kwargs) -> Optional["DatasetCollection"]:
        outputs = {}
        for name, paths in self.items():
            if isinstance(paths, ParquetPathCollection):
                outputs[name] = paths.load(transformed=transformed, **kwargs)
            else:
                id = None
                if transformed:
                    if isinstance(paths, list):
                        id = paths[0].split("/")[-1]
                    else:
                        id = paths.split("/")[-1]
                if isinstance(paths, list) or not os.path.isdir(paths):
                    outputs[name] = Dataset(paths, id=id, **kwargs)
                else:
                    outputs[name] = Dataset.from_pattern(paths, id=id, **kwargs)
                    outputs[name]._dir = paths

        if not outputs:
            return None

        return DatasetCollection(**outputs)

    @property
    def exists(self) -> bool:
        for name, paths in self.items():
            if isinstance(paths, list):
                if not all([os.path.exists(path) for path in paths]):
                    return False
            elif isinstance(paths, ParquetPathCollection):
                if not paths.exists:
                    return False
            else:
                if not os.path.exists(paths):
                    return False

        return True

    def schemas(self):
        schemas = {}
        for name, paths in self.items():
            if isinstance(paths, str) and os.path.isdir(paths):
                schemas[name] = Schema.load(paths)

        return Namespace(**schemas)

    @classmethod
    def from_splits(cls, train, eval=None, test=None):
        splits = dict(train=train)
        if eval:
            splits["eval"] = eval
        if test:
            splits["test"] = test

        return cls(**splits)

    def parquet_files(self, name):
        return glob(os.path.join(vars(self)[name], "*.parquet"))


class TabularDataset:
    def __init__(self, data_dir, client_fn=None, is_dask=False):
        self.data_dir = data_dir
        self.client_fn = client_fn
        self.transformed_dir = os.path.join(data_dir, "transformed")
        self.input_dir = os.path.join(data_dir, "orig")
        self.is_dask = is_dask
        self.transformed = None
        self.workflow = None

        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

    @abc.abstractmethod
    def create_input_column_group(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def create_default_transformations(self, prepared_paths) -> ColumnGroup:
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare(self) -> ParquetPathCollection:
        raise NotImplementedError()

    @property
    def column_group(self) -> ColumnGroup:
        return self.create_input_column_group()

    def prepare_data(self, **kwargs) -> DatasetCollection:
        LOG.info("Preparing data...")
        return self.prepare().load(transformed=False, **kwargs)

    def transformed_column_group(self, prepared_paths=None):
        if not prepared_paths:
            prepared_paths = self.prepare()
        return self.create_default_transformations(prepared_paths)

    def create_workflow(self, **kwargs):
        return Workflow(self.transformed_column_group(), self.transformed_dir)

    @contextlib.contextmanager
    def client(self):
        LOG.info("Creating Dask-client...")
        client = self.client_fn()
        try:
            yield client
        finally:
            LOG.info("Shutting down Dask-client...")
            client.shutdown()

    def statistics(
        self, transformed=False, overwrite=False, cross_columns=None, split_names=None, **kwargs
    ) -> DatasetCollectionStatistics:
        data_dir = self.transformed_dir if transformed else self.data_dir

        maybe_cached = Statistics.load(data_dir)
        if maybe_cached:
            return maybe_cached

        data = self.transform(**kwargs) if transformed else self.prepare_data()
        if split_names:
            if not isinstance(split_names, (list, tuple)):
                split_names = [split_names]
            data = data.filter_keys(*split_names)
        data = data.flatten()

        client = self.client_fn() if self.client_fn else None
        stats = data.calculate_statistics(
            data_dir, client=client, overwrite=overwrite, cross_columns=cross_columns
        )
        stats.save(data_dir)

        return stats

    def schema(self, transformed=False, **kwargs) -> Namespace:
        data_paths = self.transform_paths() if transformed else self.prepare()
        maybe_schemas = data_paths.schemas()
        if sorted(list(vars(maybe_schemas).keys())) == sorted(list(vars(data_paths).keys())):
            return maybe_schemas

        data = self.transform(**kwargs) if transformed else self.prepare_data()
        data_dir = self.transformed_dir if transformed else self.data_dir
        col_group = self.transformed_column_group(**kwargs) if transformed else self.column_group
        schemas = data.generate_schema(data_dir, col_group.tags_by_column())

        return schemas

    def transform_paths(self, workflow=None) -> ParquetPathCollection:
        prepared_paths = self.prepare()
        split_paths = prepared_paths.splits if prepared_paths.get("splits") else prepared_paths
        if not workflow:
            workflow = Workflow(self.transformed_column_group(), self.transformed_dir)

        transformed_paths = {}

        for name, paths in split_paths.items():
            if isinstance(paths, str):
                paths = split_paths.parquet_files(name)
            transformed_id = "_".join([joblib.hash(paths), workflow.column_group.id])
            transformed_paths[name] = os.path.join(self.transformed_dir, transformed_id)

        return ParquetPathCollection(**transformed_paths)

    def transform(
        self,
        workflow=None,
        overwrite=False,
        save=True,
        to_fit="train",
        for_training=False,
        **kwargs,
    ) -> Union[DatasetCollection, ParquetPathCollection]:
        client = self.client_fn() if self.client_fn else None
        if not workflow:
            workflow = Workflow(self.transformed_column_group(), self.transformed_dir)
        self.workflow = workflow

        output_paths = self.transform_paths(workflow)
        if output_paths.exists:
            LOG.info("Loading Transforming dataset from cache...")
            if for_training:
                return output_paths

            return output_paths.load(client=client, **kwargs)

        splits: DatasetCollection = self.prepare_data(client=client, **kwargs)
        splits = splits.splits if splits.get("splits") else splits

        LOG.info("Transforming dataset...")
        self.workflow.client = client
        transformed = workflow.fit_transform_collection(
            splits, to_fit=to_fit, overwrite=overwrite, save=save
        )
        if for_training:
            self.workflow.client = None
            client.shutdown()
            return output_paths

        return transformed


def create_multi_gpu_dask_client_fn(
    gpu_ids=None,
    device_limit_frac=0.7,
    # Spill GPU-Worker memory to host at this limit.
    device_pool_frac=0.8,
    part_mem_frac=0.15,
    dask_dir=None,
    protocol="tcp",
    dashboard_port="8787",
):
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)
    visible_devices = ",".join([str(n) for n in gpu_ids])

    def reinitialize_memory(self):
        reinit = lambda: rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=(device_pool_size // 256) * 256,
        )

        self.run(reinit)

    def client_fn():
        # # Check if any device memory is already occupied
        for dev in visible_devices.split(","):
            fmem = _pynvml_mem_size(kind="free", index=int(dev))
            used = (device_size - fmem) / 1e9
            if used > 1.0:
                warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

        cluster = None  # (Optional) Specify existing scheduler port
        if cluster is None:
            cluster = LocalCUDACluster(
                protocol=protocol,
                n_workers=len(visible_devices.split(",")),
                CUDA_VISIBLE_DEVICES=visible_devices,
                device_memory_limit=device_limit,
                local_directory=dask_dir,
                dashboard_address="0.0.0.0:" + str(dashboard_port),
            )

        # Create the distributed client
        client = Client(cluster)

        return client

    client_fn.part_size = part_size
    client_fn.device_pool_size = device_pool_size
    Client.reinitialize_memory = reinitialize_memory

    return client_fn
