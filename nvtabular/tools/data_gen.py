# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import random
import string

import numpy as np
import pandas as pd
import psutil

try:
    import cupy
except ImportError:
    cupy = np

try:
    import cudf
except ImportError:
    cudf = pd

from scipy import stats
from scipy.stats import powerlaw, uniform

from merlin.core.dispatch import (
    HAS_GPU,
    concat,
    create_multihot_col,
    is_list_dtype,
    make_df,
    make_series,
    pull_apart_list,
)
from merlin.core.utils import device_mem_size
from merlin.io import Dataset


class UniformDistro:
    def create_col(self, num_rows, dtype=np.float32, min_val=0, max_val=1):
        ser = make_df(np.random.uniform(min_val, max_val, size=num_rows))[0]
        ser = ser.astype(dtype)
        return ser

    def verify(self, pandas_series):
        return stats.kstest(pandas_series, uniform().cdf)


class PowerLawDistro:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def create_col(self, num_rows, dtype=np.float32, min_val=0, max_val=1):
        gamma = 1 - self.alpha
        # range 1.0 - 2.0 to avoid using 0, which represents unknown, null, None
        ser = make_df(cupy.random.uniform(0.0, 1.0, size=num_rows))[0]
        factor = cupy.power(max_val, gamma) - cupy.power(min_val, gamma)
        ser = (ser * factor.item()) + cupy.power(min_val, gamma).item()
        exp = 1.0 / gamma
        ser = ser.pow(exp)
        # replace zeroes saved for unknown
        # add in nulls if requested
        # select indexes
        return ser.astype(dtype)

    def verify(self, pandas_series):
        return stats.kstest(pandas_series, powerlaw(self.alpha).cdf)


class DatasetGen:
    def __init__(self, distribution, gpu_frac=0.8):
        """
        distribution = A Distribution Class (powerlaw, uniform)
        gpu_frac = float between 0 - 1, representing fraction of gpu
                    to use for Dataset Generation
        """
        assert distribution is not None
        self.dist = distribution
        self.gpu_frac = gpu_frac

    def create_conts(self, size, conts_rep):
        """
        size = number of rows wanted
        conts_rep = list of tuples representing values of each column
        """
        df = make_df()
        for col in conts_rep:
            dist = col.distro or self.dist
            for y in range(col.width):
                ser = dist.create_col(
                    size, min_val=col.min_val, max_val=col.max_val, dtype=col.dtype
                )
                ser.name = col.name if y == 0 else f"{col.name}_{y}"
                df = concat([df, ser], axis=1)
        return df

    def create_cats(self, size, cats_rep, entries=False):
        """
        size = number of rows
        num_cols = how many columns you want produced
        cat_rep = a list of tuple values (cardinality, min, max) representing the cardinality,
                  minimum and maximum categorical string length
        """
        # should alpha also be exposed? related to dist... should be part of that
        df = make_df()
        for col in cats_rep:
            # if mh resets size
            col_size = size
            offs = None
            dist = col.distro or self.dist
            ser = None
            if col.multi_min and col.multi_max:
                if HAS_GPU:
                    ser = dist.create_col(
                        col_size + 1, dtype=np.long, min_val=col.multi_min, max_val=col.multi_max
                    ).ceil()
                else:
                    ser = dist.create_col(
                        col_size + 1, dtype=np.long, min_val=col.multi_min, max_val=col.multi_max
                    )
                    ser = make_df(np.ceil(ser))[0]
                # sum returns numpy dtype
                col_size = int(ser.sum())
                offs = make_df(cupy.cumsum(ser.values))[0]
                offs = offs.astype("int32")
            if HAS_GPU:
                ser = dist.create_col(
                    col_size, dtype=np.long, min_val=col.min_val, max_val=col.cardinality
                ).ceil()
            else:
                ser = dist.create_col(
                    col_size, dtype=np.long, min_val=col.min_val, max_val=col.cardinality
                )
                ser = make_df(np.ceil(ser))[0]
                ser = ser.astype("int32")
            if col.permutate_index:
                ser = self.permutate_index(ser)
            if entries:
                cat_names = self.create_cat_entries(
                    col.cardinality, min_size=col.min_entry_size, max_size=col.max_entry_size
                )
                ser, _ = self.merge_cats_encoding(ser, cat_names)
            if offs is not None:
                # create multi_column from offs and ser
                ser = create_multihot_col(offs, ser)
            ser.name = col.name
            df = concat([df, ser], axis=1)
        return df

    def create_labels(self, size, labs_rep):
        df = make_df()
        for col in labs_rep:
            dist = col.distro or self.dist
            if HAS_GPU:
                ser = dist.create_col(
                    size, dtype=col.dtype, min_val=0, max_val=col.cardinality
                ).ceil()
            else:
                ser = dist.create_col(size, dtype=col.dtype, min_val=0, max_val=col.cardinality)
                ser = make_df(np.ceil(ser))[0]
            ser.name = col.name
            ser = ser.astype(col.dtype)
            df = concat([df, ser], axis=1)
        return df

    def merge_cats_encoding(self, ser, cats):
        # df and cats are both series
        # set cats to dfs
        offs = None
        if is_list_dtype(ser.dtype) or is_list_dtype(ser):
            ser, offs = pull_apart_list(ser)
        ser = make_df({"vals": ser})
        cats = make_df({"names": cats})
        cats["vals"] = cats.index
        ser = ser.merge(cats, on=["vals"], how="left")

        return ser["names"], offs

    def create_cat_entries(self, cardinality, min_size=1, max_size=5):
        set_entries = []
        while len(set_entries) <= cardinality:
            letters = string.ascii_letters + string.digits
            entry_size = random.randint(min_size, max_size)
            entry = "".join(random.choice(letters) for i in range(entry_size))
            if entry not in set_entries:
                set_entries.append(entry)
        return set_entries

    def create_df(
        self,
        size,
        cols,
        entries=False,
    ):
        conts_rep = cols["conts"] if "conts" in cols else None
        cats_rep = cols["cats"] if "cats" in cols else None
        labs_rep = cols["labels"] if "labels" in cols else None
        df = make_df()
        if conts_rep:
            df = concat([df, self.create_conts(size, conts_rep)], axis=1)
        if cats_rep:
            df = concat(
                [
                    df,
                    self.create_cats(size, cats_rep=cats_rep, entries=entries),
                ],
                axis=1,
            )
        if labs_rep:
            df = concat([df, self.create_labels(size, labs_rep)], axis=1)
        return df

    def full_df_create(
        self,
        size,
        cols,
        entries=False,
        output=".",
    ):
        files_created = []
        # always use entries for row_size estimate
        df_single = self.create_df(
            1,
            cols,
            entries=entries,
        )
        cats = cols["cats"] if "cats" in cols else None
        row_size = self.get_row_size(df_single, cats)
        batch = self.get_batch(row_size) or 1
        # ensure batch is an int
        batch = int(batch)
        file_count = 0
        while size > 0:
            x = min(batch, size)
            df = self.create_df(
                x,
                cols,
                entries=False,
            )
            full_file = os.path.join(output, f"dataset_{file_count}.parquet")
            df.to_parquet(full_file)
            files_created.append(full_file)
            size = size - x
            file_count = file_count + 1
        # rescan entire dataset to apply vocabulary to categoricals
        if entries:
            vocab_files = self.create_vocab(cats, output)
            files_created = self.merge_vocab(files_created, vocab_files, cats, output)
        # write out dataframe
        return files_created

    def create_vocab(self, cats_rep, output):
        # build vocab for necessary categoricals using cats_rep info
        vocab_files = []
        for col in cats_rep:
            col_vocab = []
            # check if needs to be split into batches for size
            batch = self.get_batch(col.cardinality * col.max_entry_size)
            file_count = 0
            completed_vocab = 0
            size = col.cardinality
            while size > completed_vocab:
                x = min(batch, size)
                # ensure stopping at desired cardinality
                x = min(x, size - completed_vocab)
                ser = make_series(
                    self.create_cat_entries(
                        x, min_size=col.min_entry_size, max_size=col.max_entry_size
                    )
                )
                # turn series to dataframe to keep index count
                ser = make_df({f"{col.name}": ser, "idx": ser.index + completed_vocab})
                file_path = os.path.join(output, f"{col.name}_vocab_{file_count}.parquet")
                completed_vocab = completed_vocab + ser.shape[0]
                file_count = file_count + 1
                # save vocab to file
                ser.to_parquet(file_path)
                col_vocab.append(file_path)
            vocab_files.append(col_vocab)
        return vocab_files

    def merge_vocab(self, dataset_file_paths, vocab_file_paths, cats_rep, output):
        # Load newly created dataset via Dataset
        final_files = []
        noe_df = Dataset(dataset_file_paths, engine="parquet")
        # for each dataframe cycle through all cat col vocabs and left merge
        for idx, df in enumerate(noe_df.to_iter()):
            for ydx, col in enumerate(cats_rep):
                # vocab_files is list of lists of file_paths
                vocab_df = Dataset(vocab_file_paths[ydx])
                for vdf in vocab_df.to_iter():
                    col_name = col.name
                    focus_col = df[col_name]
                    ser, offs = self.merge_cats_encoding(focus_col, vdf[col_name])
                    if offs is not None:
                        df[col_name] = create_multihot_col(offs, ser)
                    else:
                        df[col_name] = ser
            file_path = os.path.join(output, f"FINAL_{idx}.parquet")
            df.to_parquet(file_path)
            final_files.append(file_path)
        return final_files

    def verify_df(self, df_to_verify):
        sts = []
        ps = []
        for col in df_to_verify.columns:
            if HAS_GPU:
                st_df, p_df = self.dist.verify(df_to_verify[col].to_pandas())
            else:
                st_df, p_df = self.dist.verify(df_to_verify[col])
            sts.append(st_df)
            ps.append(p_df)
        return sts, ps

    def get_batch(self, row_size):
        # grab max amount of gpu memory
        if HAS_GPU:
            mem = device_mem_size(kind="total") * self.gpu_frac
        else:
            mem = psutil.virtual_memory().available / 1024

        # find # of rows fit in gpu memory
        return mem // row_size

    def get_row_size(self, row, cats_rep):
        """
        row = cudf.DataFrame comprising of one row
        """
        size = 0
        for col in row.columns:
            if is_list_dtype(row[col].dtype):
                # second from last position is max list length
                # find correct cats_rep by scanning through all for column name
                tar = self.find_target_rep(col, cats_rep)
                # else use default 1
                val = tar.multi_max if tar else 1
                size = size + row[col]._column.elements.dtype.itemsize * val
            else:
                size = size + row[col].dtype.itemsize
        return size

    def find_target_rep(self, name, cats_rep):
        for rep in cats_rep:
            if name == rep.name:
                return rep
        return None

    def permutate_index(self, ser):
        name = ser.name
        ser.name = "ind"
        ind = ser.drop_duplicates().values
        ind_random = cupy.random.permutation(ind)
        df_map = cudf.DataFrame({"ind": ind, "ind_random": ind_random})
        if not HAS_GPU:
            ser = cudf.DataFrame(ser)
        ser = ser.merge(df_map, how="left", left_on="ind", right_on="ind")["ind_random"]
        ser.name = name
        return ser


DISTRO_TYPES = {"powerlaw": PowerLawDistro, "uniform": UniformDistro}


class Col:
    def __init__(self, name, dtype, distro=None):
        self.name = name
        self.dtype = dtype
        self.distro = distro


class ContCol(Col):
    def __init__(
        self,
        name,
        dtype,
        min_val=0,
        max_val=1,
        mean=None,
        std=None,
        per_nan=None,
        width=1,
        distro=None,
    ):
        super().__init__(name, dtype, distro)
        self.min_val = min_val
        self.max_val = max_val
        self.mean = mean
        self.std = std
        self.per_nan = per_nan
        self.width = width


class CatCol(Col):
    def __init__(
        self,
        name,
        dtype,
        cardinality,
        min_entry_size=None,
        max_entry_size=None,
        avg_entry_size=None,
        per_nan=None,
        multi_min=None,
        multi_max=None,
        multi_avg=None,
        distro=None,
        min_val=0,
        permutate_index=False,
    ):
        super().__init__(name, dtype, distro)
        self.cardinality = cardinality
        self.min_entry_size = min_entry_size
        self.max_entry_size = max_entry_size
        self.avg_entry_size = avg_entry_size
        self.per_nan = per_nan
        self.multi_min = multi_min
        self.multi_max = multi_max
        self.multi_avg = multi_avg
        self.min_val = min_val
        self.permutate_index = permutate_index


class LabelCol(Col):
    def __init__(self, name, dtype, cardinality, per_nan=None, distro=None):
        super().__init__(name, dtype, distro)
        self.cardinality = cardinality
        self.per_nan = per_nan


def _get_cols_from_schema(schema, distros=None):
    """
    schema = a dictionary comprised of column information,
             where keys are column names, and the value
             contains spec info about column.

    Schema example

    num_rows:
    conts:
        col_name:
            dtype:
            min_val:
            max_val:
            mean:
            std:
            per_nan:
    cats:
        col_name:
            dtype:
            cardinality:
            min_entry_size:
            max_entry_size:
            avg_entry_size:
            per_nan:
            multi_min:
            multi_max:
            multi_avg:
            min_val:
            permutate_index:

    labels:
        col_name:
            dtype:
            cardinality:
            per_nan:
    """
    cols = {}
    executor = {"conts": ContCol, "cats": CatCol, "labels": LabelCol}
    for section, vals in schema.items():
        if section == "num_rows":
            continue
        cols[section] = []
        for col_name, val in vals.items():
            v_dict = {"name": col_name}
            v_dict.update(val)
            if distros and col_name in distros:
                dis = distros[col_name]
                new_distr = DISTRO_TYPES[dis["name"]](**dis["params"])
                v_dict.update({"distro": new_distr})
            cols[section].append(executor[section](**v_dict))
    return cols
