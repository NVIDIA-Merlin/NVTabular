#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
import uuid

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import rmm
from cudf.utils.dtypes import min_scalar_type

import nvtabular.io


class DLLabelEncoder(object):
    """
    This is the class that Encoder uses to
    transform categoricals to unique integer values.

    Parameters
    -----------
    col : str
        column name
    cats : list
        pre-calculated unique values.
    path : str
        path to store unique values if needed.
    use_frequency : bool
        use frequency based transformation or not.
    freq_threshold : int, default 0
        threshold value for frequency based transformation.
    limit_frac : float, default 0.1
        fraction of memory to use during unique id calculation.
    gpu_mem_util_limit : float, default 0.8
        GPU memory utilization limit during frequency based
        calculation. If limit is exceeded, unique ids are moved
        to host memory.
    gpu_mem_trans_use : float, default 0.8
        GPU memory utilization limit during transformation. How much
        GPU memory will be used during transformation is calculated
        using this parameter.
    file_paths : list of str
        path(s) of the files that unique ids are stored.

    """

    def __init__(
        self,
        col,
        cats=None,
        path=None,
        use_frequency=False,
        freq_threshold=0,
        limit_frac=0.1,
        gpu_mem_util_limit=0.1,
        cpu_mem_util_limit=0.1,
        gpu_mem_trans_use=0.1,
        file_paths=None,
    ):

        if freq_threshold < 0:
            raise ValueError("freq_threshold cannot be lower than 1.")

        if limit_frac < 0.0 or limit_frac > 1.0:
            raise ValueError("limit_frac has to be between 0 and 1.")

        if gpu_mem_util_limit < 0.0 or gpu_mem_util_limit > 1.0:
            raise ValueError("gpu_mem_util_limit has to be between 0 and 1.")

        if gpu_mem_trans_use < 0.0 or gpu_mem_trans_use > 1.0:
            raise ValueError("gpu_mem_trans_use has to be between 0 and 1.")

        self._cats_counts = cudf.Series([])
        self._cats_counts_host = None
        self._cats_host = None
        self._cats_parts = []
        self._cats_host = cats.to_pandas() if type(cats) == cudf.Series else cats
        self.path = path or os.path.join(os.getcwd(), "label_encoders")
        self.folder_path = os.path.join(self.path, col)
        self.file_paths = file_paths or []
        self.ignore_files = []
        if os.path.exists(self.folder_path):
            self.ignore_files = [
                os.path.join(self.folder_path, x)
                for x in os.listdir(self.folder_path)
                if x.endswith("parquet") and x not in self.file_paths
            ]
        self.col = col
        self.use_frequency = use_frequency
        self.freq_threshold = freq_threshold
        self.limit_frac = limit_frac
        self.gpu_mem_util_limit = gpu_mem_util_limit
        self.gpu_mem_trans_use = gpu_mem_trans_use
        self.cat_exp_count = 0

    def _label_encoding(self, vals, cats, dtype=None, na_sentinel=-1):
        if dtype is None:
            dtype = min_scalar_type(len(cats), 32)

        order = cudf.Series(cp.arange(len(vals)))
        codes = cats.index

        value = cudf.DataFrame({"value": cats, "code": codes})
        codes = cudf.DataFrame({"value": vals.copy(), "order": order})
        codes = codes.merge(value, on="value", how="left")
        codes = codes.sort_values("order")["code"]
        codes.fillna(na_sentinel, inplace=True)
        cats.name = None  # because it was mutated above
        return codes._copy_construct(name=None, index=vals.index)

    def transform(self, y: cudf.Series, unk_idx=0) -> cudf.Series:
        """
        Maps y to unique ids.

        Parameters
        -----------
        y : cudf Series

        Returns
        -----------
        encoded: cudf Series
        """

        if self._cats_host is None:
            raise Exception("Encoder was not fit!")

        if len(self._cats_host) == 0:
            raise Exception("Encoder was not fit!")

        avail_gpu_mem = rmm.get_info().free
        sub_cats_size = int(avail_gpu_mem * self.gpu_mem_trans_use / self._cats_host.dtype.itemsize)
        i = 0
        encoded = None
        while i < len(self._cats_host):
            sub_cats = cudf.Series(self._cats_host[i : i + sub_cats_size])
            if encoded is None:
                encoded = self._label_encoding(y, sub_cats, na_sentinel=0)
            else:
                encoded = encoded.add(
                    self._label_encoding(y, sub_cats, na_sentinel=0), fill_value=0,
                )
            i = i + sub_cats_size

        sub_cats = cudf.Series([])
        return encoded[:].replace(-1, 0)

    def _series_size(self, s):
        if hasattr(s, "str"):
            return s.str.device_memory()
        else:
            return s.dtype.itemsize * len(s)

    # Returns GPU available space and utilization.
    def _get_gpu_mem_info(self):
        gpu_free_mem, gpu_total_mem = rmm.get_info()
        gpu_mem_util = (gpu_total_mem - gpu_free_mem) / gpu_total_mem
        return gpu_free_mem, gpu_mem_util

    def fit(self, y: cudf.Series):
        """
        Calculates unique values or value counts of y
        and moves them to host memory.

        Parameters
        -----------
        y : cudf Series
        """

        if self.use_frequency:
            self._fit_freq(y)
        else:
            self._fit_unique(y)

    def fit_finalize(self):
        """
        Finalizes the fit operation. Gets unique values or
        value counts from host memory and merge them into one
        big array.
        """

        if self.use_frequency:
            return self._fit_freq_finalize()
        else:
            return self._fit_unique_finalize()

    def _fit_unique(self, y: cudf.Series):
        y_uniqs = y.unique()
        self._cats_parts.append(y_uniqs.to_pandas())

    def _fit_unique_finalize(self):
        y_uniqs = cudf.Series([]) if self._cats_host is None else cudf.from_pandas(self._cats_host)
        for i in range(len(self._cats_parts)):
            y_uniqs_part = cudf.from_pandas(self._cats_parts.pop())
            if y_uniqs.shape[0] == 0:
                y_uniqs = y_uniqs_part
            else:
                y_uniqs = y_uniqs.append(y_uniqs_part).unique()  # Check merge option as well

        # Can't just pass None as a placeholder, since that automatically gets converted
        # to -1 later (cudf.Series([None]).to_pandas() == (-1,)) for int columns.
        # Instead come up with a suitable default
        # https://github.com/rapidsai/recsys/issues/73
        na_value = _get_na_value(y_uniqs.dtype)

        cats = cudf.Series([na_value]).append(y_uniqs).astype(y_uniqs.dtype)
        cats = cats.unique()
        cats.reset_index(drop=True, inplace=True)
        self._cats_host = cats.to_pandas()
        return self._cats_host.shape[0]

    def _fit_freq(self, y: cudf.Series):
        y_counts = y.value_counts()
        self._cats_parts.append(y_counts.to_pandas())

    def _fit_freq_finalize(self):
        y_counts = cudf.Series([])
        cats_counts_host = []
        for i in range(len(self._cats_parts)):
            y_counts_part = cudf.from_pandas(self._cats_parts.pop())
            if y_counts.shape[0] == 0:
                y_counts = y_counts_part
            else:
                y_counts = y_counts.add(y_counts_part, fill_value=0)
            _series_size_gpu = self._series_size(y_counts)

            avail_gpu_mem, gpu_mem_util = self._get_gpu_mem_info()
            if (
                _series_size_gpu > (avail_gpu_mem * self.limit_frac)
                or gpu_mem_util > self.gpu_mem_util_limit
            ):
                cats_counts_host.append(y_counts.to_pandas())
                y_counts = cudf.Series([])

        if len(cats_counts_host) == 0:
            cats = cudf.Series(y_counts[y_counts >= self.freq_threshold].index)
            cats = cudf.Series([None]).append(cats)
            cats.reset_index(drop=True, inplace=True)
            self._cats_host = cats.to_pandas()
        else:
            y_counts_host = cats_counts_host.pop()
            for i in range(len(cats_counts_host)):
                y_counts_host_temp = cats_counts_host.pop()
                y_counts_host = y_counts_host.add(y_counts_host_temp, fill_value=0)

            self._cats_host = pd.Series(y_counts_host[y_counts_host >= self.freq_threshold].index)
            self._cats_host = pd.Series([None]).append(self._cats_host)
            self._cats_host.reset_index(drop=True, inplace=True)

        return self._cats_host.shape[0]

    def merge_series(self, compr_a, compr_b):
        df, dg = cudf.DataFrame(), cudf.DataFrame()
        df["l1"] = compr_a.nans_to_nulls().dropna()
        dg["l2"] = compr_b.nans_to_nulls().dropna()
        mask = dg["l2"].isin(df["l1"])
        unis = dg.loc[~mask]["l2"].unique()
        return unis

    def dump_cats(self):
        x = cudf.DataFrame()
        x[self.col] = self._cats.unique()
        self.cat_exp_count = self.cat_exp_count + x.shape[0]
        file_id = str(uuid.uuid4().hex) + ".parquet"
        tar_file = os.path.join(self.folder_path, file_id)
        x.to_parquet(tar_file, compression=None)
        self._cats = cudf.Series()
        # should find new file just exported
        new_file_path = [
            os.path.join(self.folder_path, x)
            for x in os.listdir(self.folder_path)
            if x.endswith("parquet") and x not in self.file_paths and x not in self.ignore_files
        ]
        # add file to list
        self.file_paths.extend(new_file_path)
        self.file_paths = list(set(self.file_paths))

    def one_cycle(self, compr):
        # compr is already a list of unique values to check against
        if os.path.exists(self.folder_path):
            file_paths = [
                os.path.join(self.folder_path, x)
                for x in os.listdir(self.folder_path)
                if x.endswith("parquet") and x not in self.file_paths + self.ignore_files
            ]
            if file_paths:
                chunks = nvtabular.io.GPUDatasetIterator(file_paths)
                for chunk in chunks:
                    compr = self.merge_series(chunk[self.col], compr)
                    if len(compr) == 0:
                        # if nothing is left to compare... bug out
                        break
        return compr

    def get_cats(self):
        gdf = cudf.from_pandas(self._cats_host)
        gdf.reset_index(drop=True, inplace=True)
        return gdf

    def __repr__(self):
        return "{0}(_cats={1!r})".format(type(self).__name__, self.get_cats().values_to_string())


def _get_na_value(dtype):
    """ Returns a suitable value for missing values based off the dtype of the col """
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min

    elif np.issubdtype(dtype, np.generic):
        return None

    else:
        raise ValueError(f"unhandled dtype for encoder: '{dtype}'")
