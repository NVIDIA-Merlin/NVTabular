import os
from typing import Any, Optional

import pandas as pd

from nvtabular.io.dataset import LOG, Dataset
from nvtabular.utils import Namespace


class DatasetCollection(Namespace):
    def __init__(self, **kwargs: Any) -> None:
        splits = {}
        for key, val in kwargs.items():
            if not val:
                LOG.warning("%s is emtpy, remove it from the collection", key)
            else:
                splits[key] = val
        assert all(isinstance(dataset, (Dataset, DatasetCollection)) for dataset in kwargs.values())
        super().__init__(**kwargs)

    @classmethod
    def from_splits(cls, train, eval=None, test=None):  # pylint: disable=W0622
        splits = dict(train=train)
        if eval:
            splits["eval"] = eval
        if test:
            splits["test"] = test
        return cls(**splits)

    def filter_keys(self, *keys):
        outputs = {}
        for name, dataset in self.items():
            if name in keys:
                outputs[name] = dataset

        return self.__class__(**outputs)

    def flatten(self):
        outputs = self.to_dict(flattened=True)

        return self.__class__(**outputs)

    def to_dict(self, flattened=False):
        outputs = {}
        for name, dataset in self.items():
            if isinstance(dataset, DatasetCollection):
                outputs[name] = dataset.to_dict(flattened=flattened)
            else:
                outputs[name] = dataset

        if flattened:
            df = pd.json_normalize(outputs, sep="/")
            outputs = df.to_dict(orient="records")[0]

        return outputs

    def to_parquet(
        self,
        output_path,
        by_id=True,
        overwrite=False,
        shuffle=None,
        preserve_files=False,
        output_files=None,
        out_files_per_proc=None,
        num_threads=0,
        dtypes=None,
        cats=None,
        conts=None,
        labels=None,
        suffix=".parquet",
        partition_on=None,
    ):
        kwargs = locals()
        by_id, overwrite = kwargs.pop("by_id"), kwargs.pop("overwrite")
        for key in ["self", "output_path"]:
            del kwargs[key]

        for name, dataset in self.items():
            dataset_dir = os.path.join(output_path, dataset.id if by_id else name)
            dataset._dir = dataset_dir
            if not os.path.exists(dataset_dir) or overwrite:
                LOG.info("Saving to %s", dataset_dir)
                dataset.to_parquet(dataset_dir, **kwargs)

        return self

    @classmethod
    def load_from_path(
        cls, path, split_dict, data_format="parquet"
    ) -> Optional["DatasetCollection"]:
        outputs = {}

        for name, dataset_id in split_dict.items():
            dataset_path = os.path.join(path, dataset_id)
            if os.path.exists(path):
                outputs[name] = Dataset.from_pattern(dataset_path, f"*.{data_format}")

        if not outputs:
            return None

        return DatasetCollection(**outputs)
