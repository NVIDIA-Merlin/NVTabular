import os

import abc
import warnings

import rmm

from ..column_group import ColumnGroup
from ..io.dataset import DatasetCollection
from ..workflow import Workflow
from nvtabular.utils import device_mem_size, _pynvml_mem_size

from dask.distributed import Client
from dask_cuda import LocalCUDACluster


class TabularDataset:
    def __init__(self, data_dir, client=None, is_dask=False):
        self.data_dir = data_dir
        self.client = client
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
    def create_default_transformations(self, data) -> ColumnGroup:
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare(self, **kwargs) -> DatasetCollection:
        raise NotImplementedError()

    @property
    def column_group(self) -> ColumnGroup:
        return self.create_input_column_group()

    @property
    def data(self):
        return self.prepare()

    def transform(self, workflow=None, overwrite=False, save=True, to_fit="train",
                  for_training=False, **kwargs) -> DatasetCollection:
        splits = self.prepare(**kwargs)
        if not workflow:
            workflow = Workflow(self.create_default_transformations(splits), self.transformed_dir,
                                client=self.client)
        self.workflow = workflow

        splits = splits.splits if splits.get("splits") else splits

        splits = workflow.fit_transform_collection(splits, to_fit=to_fit, overwrite=overwrite, save=save)
        self.transformed = splits

        if for_training:
            return self.release_memory_for_training()

        return splits

    def release_memory_for_training(self):
        if self.client:
            self.client.restart()
            self.client.close()

        return self.transform()


def create_multi_gpu_dask_client(gpu_ids=None,
                                 device_limit_frac=0.7,  # Spill GPU-Worker memory to host at this limit.
                                 device_pool_frac=0.8,
                                 part_mem_frac=0.15,
                                 dask_dir=None,
                                 protocol="tcp",
                                 dashboard_port="8787"):
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)
    visible_devices = ",".join([str(n) for n in gpu_ids])

    # # Check if any device memory is already occupied
    # for dev in visible_devices.split(","):
    #     fmem = _pynvml_mem_size(kind="free", index=int(dev))
    #     used = (device_size - fmem) / 1e9
    #     if used > 1.0:
    #         warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

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

    def reinitialize_memory(self):
        reinit = lambda: rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=(device_pool_size // 256) * 256,
        )

        self.run(reinit)

    # Create the distributed client
    client = Client(cluster)
    client.part_size = part_size
    client.device_pool_size = device_pool_size
    Client.reinitialize_memory = reinitialize_memory

    return client
