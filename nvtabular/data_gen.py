import os
import random
import string

import cudf
import cupy
import numpy as np
from cudf.core.column import as_column, build_column
from scipy import stats
from scipy.stats import powerlaw, uniform

from .io.dataset import Dataset
from .utils import device_mem_size


class UniformDistro:
    def create_col(self, num_rows, dtype=np.float32, min_val=0, max_val=1):
        ser = cudf.Series(cupy.random.uniform(min_val, max_val, size=num_rows))
        return ser

    def verify(self, pandas_series):
        return stats.kstest(pandas_series, uniform().cdf)


class PowerLawDistro:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def create_col(self, num_rows, dtype=np.float32, min_val=0, max_val=1):
        gamma = 1 - self.alpha
        # range 1.0 - 2.0 to avoid using 0, which represents unknown, null, None
        ser = cudf.Series(cupy.random.uniform(1.0, 2.0, size=num_rows))
        factor = (cupy.power(max_val, gamma) - cupy.power(min_val, gamma)) + cupy.power(
            min_val, gamma
        )
        ser = ser * factor.item()
        exp = 1.0 / gamma
        ser = ser.pow(exp)
        # halving to account for 1.0 - 2.0 range
        ser = ser // 2
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

    # need to be able to generate from scratch
    def create_conts(self, size, conts_rep):
        """
        size = number of rows wanted
        conts_rep = list of tuples representing values of each column
        """
        num_cols = len(conts_rep)
        df = cudf.DataFrame()
        for x in range(num_cols):
            dtype, min_val, max_val = conts_rep[x][0:3]
            ser = self.dist.create_col(size, min_val=min_val, max_val=max_val)
            ser = ser.astype(dtype)
            ser.name = f"CONT_{x}"
            df = cudf.concat([df, ser], axis=1)
        return df

    def create_cats(self, size, cats_rep, entries=False):
        """
        size = number of rows
        num_cols = how many columns you want produced
        cat_rep = a list of tuple values (cardinality, min, max) representing the cardinality,
                  minimum and maximum categorical string length
        """
        # should alpha also be exposed? related to dist... should be part of that
        num_cols = len(cats_rep)
        df = cudf.DataFrame()
        for x in range(num_cols):
            # if mh resets size
            col_size = size
            offs = None
            cardinality, minn, maxx = cats_rep[x][1:4]
            # calculate number of elements in each row for num rows
            mh_min, mh_max, mh_avg = cats_rep[x][6:]
            if mh_min and mh_max:
                entrys_lens = self.dist.create_col(
                    col_size + 1, dtype=np.long, min_val=mh_min, max_val=mh_max
                ).ceil()
                # sum returns numpy dtype
                col_size = int(entrys_lens.sum())
                offs = cupy.cumsum(entrys_lens.values)
            ser = self.dist.create_col(
                col_size, dtype=np.long, min_val=0, max_val=cardinality
            ).ceil()
            if entries:
                cat_names = self.create_cat_entries(cardinality, min_size=minn, max_size=maxx)
                ser, _ = self.merge_cats_encoding(ser, cat_names)
            if offs is not None:
                # create multi_column from offs and ser
                ser = self.create_multihot_col(offs, ser)
            ser.name = f"CAT_{x}"
            df = cudf.concat([df, ser], axis=1)
        return df

    def create_labels(self, size, labs_rep):
        num_cols = len(labs_rep)
        df = cudf.DataFrame()
        for x in range(num_cols):
            cardinality = labs_rep[x][1]
            ser = self.dist.create_col(size, dtype=np.long, min_val=1, max_val=cardinality).ceil()
            # bring back down to correct representation because of ceil call
            ser = ser - 1
            ser.name = f"LAB_{x}"
            df = cudf.concat([df, ser], axis=1)
        return df

    def merge_cats_encoding(self, ser, cats):
        # df and cats are both series
        # set cats to dfs
        offs = None
        if type(ser.dtype) is cudf.core.dtypes.ListDtype:
            offs = ser._column.offsets
            ser = ser.list.leaves
        ser = cudf.DataFrame({"vals": ser})
        cats = cudf.DataFrame({"names": cats})
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

    def create_multihot_col(self, offsets, data):
        """
        offsets = cudf series with offset values for list data
        data = cudf series with the list data flattened to 1-d
        """
        offs = as_column(offsets, dtype="int32")
        encoded = as_column(data)
        col = build_column(
            None,
            size=offs.size - 1,
            dtype=cudf.core.dtypes.ListDtype(encoded.dtype),
            children=(offs, encoded),
        )
        return cudf.Series(col)

    def create_df(
        self,
        size,
        conts_rep=None,
        cats_rep=None,
        labs_rep=None,
        dist=PowerLawDistro(),
        entries=False,
    ):
        df = cudf.DataFrame()
        if conts_rep:
            df = cudf.concat([df, self.create_conts(size, conts_rep)], axis=1)
        if cats_rep:
            df = cudf.concat(
                [
                    df,
                    self.create_cats(size, cats_rep=cats_rep, entries=entries),
                ],
                axis=1,
            )
        if labs_rep:
            df = cudf.concat([df, self.create_labels(size, labs_rep)], axis=1)
        return df

    def full_df_create(
        self,
        size,
        conts_rep=None,
        cats_rep=None,
        labs_rep=None,
        dist=PowerLawDistro(),
        entries=False,
        output=".",
    ):
        files_created = []
        # always use entries for row_size estimate
        df_single = self.create_df(
            1,
            conts_rep=conts_rep,
            cats_rep=cats_rep,
            labs_rep=labs_rep,
            dist=dist,
            entries=entries,
        )
        row_size = self.get_row_size(df_single, cats_rep)
        batch = self.get_batch(row_size)
        file_count = 0
        while size > 0:
            x = batch
            if size < batch:
                x = size
            df = self.create_df(
                x,
                conts_rep=conts_rep,
                cats_rep=cats_rep,
                labs_rep=labs_rep,
                dist=dist,
                entries=False,
            )
            full_file = os.path.join(output, f"dataset_{file_count}.parquet")
            df.to_parquet(full_file)
            files_created.append(full_file)
            size = size - batch
            file_count = file_count + 1
        # rescan entire dataset to apply vocabulary to categoricals
        if entries:
            vocab_files = self.create_vocab(cats_rep, output)
            files_created = self.merge_vocab(files_created, vocab_files, cats_rep, output)
        # write out dataframe
        return files_created

    def create_vocab(self, cats_rep, output):
        # build vocab for necessary categoricals using cats_rep info
        vocab_files = []
        for idx, cat_rep in enumerate(cats_rep):
            col_vocab = []
            cardinality, minn, maxx = cat_rep[1:4]
            # check if needs to be split into batches for size
            batch = self.get_batch(cardinality * maxx)
            file_count = 0
            completed_vocab = 0
            size = cardinality
            while size > completed_vocab:
                x = batch
                if size < batch:
                    x = size
                ser = cudf.Series(self.create_cat_entries(x, min_size=minn, max_size=maxx))
                # turn series to dataframe to keep index count
                ser = cudf.DataFrame({f"CAT_{idx}": ser, "idx": ser.index + completed_vocab})
                file_path = os.path.join(output, f"CAT_{idx}_vocab_{file_count}.parquet")
                completed_vocab = completed_vocab + x
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
            for x in range(0, len(cats_rep)):
                # vocab_files is list of lists of file_paths
                vocab_df = Dataset(vocab_file_paths[x])
                for vdf in vocab_df.to_iter():
                    col_name = f"CAT_{x}"
                    focus_col = df[col_name]
                    ser, offs = self.merge_cats_encoding(focus_col, vdf[col_name])
                    if offs:
                        df[col_name] = self.create_multihot_col(offs, ser)
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
            st_df, p_df = self.dist.verify(df_to_verify[col].to_pandas())
            sts.append(st_df)
            ps.append(p_df)
        return sts, ps

    def get_batch(self, row_size):
        # grab max amount of gpu memory
        gpu_mem = device_mem_size(kind="free") * self.gpu_frac
        # find # of rows fit in gpu memory
        return gpu_mem // row_size

    def get_row_size(self, row, cats_reps):
        """
        row = cudf.DataFrame comprising of one row
        """
        size = 0
        for col in row.columns:
            if "CAT" in col:
                if type(row[col].dtype) is cudf.core.dtypes.ListDtype:
                    # column names for categorical CAT_x
                    val_num = int(col.split("_")[1])
                    # second from last position is max list length
                    max_size = cats_reps[val_num][-2]
                    size = size + row[col]._column.elements.dtype.itemsize * max_size
            else:
                size = size + row[col].dtype.itemsize
        return size


class Col:
    def tupel(self):
        tupel = []
        for attr, val in self.__dict__.items():
            tupel.append(val)
        return tupel


class ContCol(Col):
    def __init__(self, dtype, min_val=0, max_val=1, mean=None, std=None, per_nan=None):
        self.dtype = dtype
        self.min_val = min_val
        self.max_val = max_val
        self.mean = mean
        self.std = std
        self.per_nan = per_nan


class CatCol(Col):
    def __init__(
        self,
        dtype,
        cardinality,
        min_entry_size=None,
        max_entry_size=None,
        avg_entry_size=None,
        per_nan=None,
        multi_min=None,
        multi_max=None,
        multi_avg=None,
    ):
        self.dtype = dtype
        self.cardinality = cardinality
        self.min_entry_size = min_entry_size
        self.max_entry_size = max_entry_size
        self.avg_entry_size = avg_entry_size
        self.per_nan = None
        self.multi_min = multi_min
        self.multi_max = multi_max
        self.multi_avg = multi_avg


class LabelCol(Col):
    def __init__(self, dtype, cardinality):
        self.dtype = dtype
        self.cardinality = cardinality


def _get_cols_from_schema(schema):
    """
    schema = a dictionary comprised of column information,
             where keys are column names, and the value
             contains spec info about column.

    Schema example

    conts:
        col_name:
            dtype:
            min_val:
            max_val:
            mean:
            standard deviation:
            % NaNs:
    cats:
        col_name:
            dtype:
            cardinality:
            min_entry_size:
            max_entry_size:
            avg_entry_size:
            % NaNs:
            multi_min:
            multi_max:
            multi_avg:

    labels:
        col_name:
            dtype:
            cardinality:
            % NaNs:
    """
    cols = {}
    executor = {"conts": ContCol, "cats": CatCol, "labs": LabelCol}
    for section, vals in schema.items():
        cols[section] = []
        for col_name, val in vals.items():
            cols[section].append(executor[section](**val).tupel())
    return cols
