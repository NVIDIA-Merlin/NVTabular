import logging
import os

import numpy as np

from nvtabular import ops
from nvtabular.column_group import ColumnGroup, Tag
from nvtabular.dataset.base import ParquetPathCollection, TabularDataset
from nvtabular.io import Dataset
from nvtabular.io.dataset import DatasetCollection
from nvtabular.utils import download_file

LOG = logging.getLogger("nvtabular")


class Criteo(TabularDataset):
    def __init__(self, work_dir, num_days=2, client_fn=None):
        super().__init__(os.path.join(work_dir, self.name()), client_fn=client_fn)
        self.num_days = num_days
        self.parquet_dir = os.path.join(self.input_dir, "parquet")

    def create_input_column_group(self):
        outputs = ColumnGroup(["I" + str(x) for x in range(1, 14)], tags=Tag.CONTINUOUS)
        outputs += ColumnGroup(["C" + str(x) for x in range(1, 27)], tags=Tag.CATEGORICAL)
        outputs += ColumnGroup(["label"], tags=Tag.TARGETS_BINARY)

        return outputs

    def create_default_transformations(self, data) -> ColumnGroup:
        outputs = self.column_group.targets_column_group
        outputs += self.column_group.categorical_column_group >> ops.Categorify(
            out_path=os.path.join(self.data_dir, "categories"), max_size=10000000
        )
        outputs += (
            self.column_group.continuous_column_group
            >> ops.FillMissing()
            >> ops.Clip(min_value=0)
            >> ops.Normalize()
        )

        return outputs

    def convert_files_to_parquet(self, filenames, frac_size=0.10):
        cont_names = ["I" + str(x) for x in range(1, 14)]
        cat_names = ["C" + str(x) for x in range(1, 27)]
        cols = ["label"] + cont_names + cat_names

        # Specify column dtypes. Note that "hex" means that
        # the values will be hexadecimal strings that should
        # be converted to int32
        dtypes = {}
        dtypes["label"] = np.int32
        for x in cont_names:
            dtypes[x] = np.int32
        for x in cat_names:
            dtypes[x] = "hex"

        # Create an NVTabular Dataset from a CSV-file glob
        dataset = Dataset(
            filenames,
            engine="csv",
            names=cols,
            part_mem_fraction=frac_size,
            sep="\t",
            dtypes=dtypes,
            client=self.client,
        )

        dataset.to_parquet(self.parquet_dir, preserve_files=True)

    def prepare(self) -> ParquetPathCollection:
        orig_files = []
        # Iterate over days
        for i in range(0, self.num_days):
            file = os.path.join(self.input_dir, "day_" + str(i) + ".gz")
            orig_files.append(file.replace(".gz", ""))
            # Download file, if there is no .gz, .csv or .parquet file
            if not (
                os.path.exists(file)
                or os.path.exists(file.replace(".gz", ".parquet").replace("orig", "orig/parquet/"))
                or os.path.exists(file.replace(".gz", ""))
            ):
                download_file(
                    "http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_"
                    + str(i)
                    + ".gz",
                    file,
                )

        # Check if files need to be converted
        files_to_convert = []
        for f in orig_files:
            parquet_file = f"{f}.parquet".replace("orig", "orig/parquet")
            if not os.path.exists(parquet_file):
                files_to_convert.append(f)
        if files_to_convert:
            self.convert_files_to_parquet(files_to_convert)

        parquet_files = [
            os.path.join(self.parquet_dir, f"day_{i}.parquet") for i in range(0, self.num_days)
        ]

        return ParquetPathCollection.from_splits(parquet_files[:-1], eval=parquet_files[-1])

    def name(self):
        return f"criteo"
