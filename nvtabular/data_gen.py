import random
import string

import cudf
import cupy
import numpy as np
from scipy import stats
from scipy.stats import powerlaw, uniform

#from .utils import device_mem_size


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
        ser = cudf.Series(cupy.random.uniform(0.0, 1.0, size=num_rows))
        factor = (cupy.power(max_val, gamma) - cupy.power(min_val, gamma)) + cupy.power(
            min_val, gamma
        )
        ser = ser * factor.item()
        exp = 1.0 / gamma
        ser = ser.pow(exp)
        # add in nulls if requested
        # select indexes
        return ser.astype(dtype)

    def verify(self, pandas_series):
        return stats.kstest(pandas_series, powerlaw(self.alpha).cdf)


class DatasetGen:
    def __init__(self, distribution):
        assert distribution is not None
        self.dist = distribution

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

        offs = None
        num_cols = len(cats_rep)
        df = cudf.DataFrame()
        for x in range(num_cols):
            cardinality, minn, maxx = cats_rep[x][1:4]
            # calculate number of elements in each row for num rows
            mh_min, mh_max, mh_avg = cats_rep[x][6:]
            if mh_min and mh_max:
                entrys_lens = self.dist.create_col(
                    size, dtype=np.long, min_val=mh_min, max_val=mh_max
                ).ceil()
                size = entrys_lens.sum()
                offs = cupy.cumsum(entrys_lens)
            ser = self.dist.create_col(size, dtype=np.long, min_val=1.0, max_val=cardinality).ceil()
            if entries:
                cat_names = self.create_cat_entries(cardinality, min_size=minn, max_size=maxx)
                ser = self.merge_cats_encoding(ser, cat_names)
            ser.name = f"CAT_{x}"
            if offs:
                # create multi_column from offs and ser
                # TODO: remove stub
                pass
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
        ser = cudf.DataFrame({"vals": ser})
        cats = cudf.DataFrame({"names": cats})
        cats["vals"] = cats.index
        ser = ser.merge(cats, on=["vals"], how="left")
        return ser["names"]

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

    def verify_df(self, df_to_verify):
        sts = []
        ps = []
        for col in df_to_verify.columns:
            st_df, p_df = self.dist.verify(df_to_verify[col].to_pandas())
            sts.append(st_df)
            ps.append(p_df)
        return sts, ps


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
        max_entry_size=None,
        min_entry_size=None,
        avg_entry_size=None,
        per_nan=None,
        multi_avg=None,
        multi_min=None,
        multi_max=None,
    ):
        self.dtype = dtype
        self.cardinality = cardinality
        self.max_entry_size = max_entry_size
        self.min_entry_size = min_entry_size
        self.avg_entry_size = avg_entry_size
        self.per_nan = None
        self.multi_avg = multi_avg
        self.multi_min = multi_min
        self.multi_max = multi_max


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
            max_entry_size:
            min_entry_size:
            avg_entry_size:
            % NaNs:
            multi_avg:
            multi_min:
            multi_max:
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
