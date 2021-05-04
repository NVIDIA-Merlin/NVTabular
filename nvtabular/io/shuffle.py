#
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
import enum
import warnings

import cupy as cp
import numpy as np
import pandas as pd


class Shuffle(enum.Enum):
    PER_PARTITION = 0
    PER_WORKER = 1
    FULL = 2


#
# Helper Function definitions
#


def _check_shuffle_arg(shuffle):
    if shuffle is None:
        return shuffle

    if isinstance(shuffle, Shuffle):
        if shuffle == Shuffle.FULL:
            raise ValueError('`shuffle="full"` is not yet supported.')
    elif shuffle is True:
        shuffle = Shuffle.PER_WORKER
        warnings.warn("`shuffle=True` is deprecated. Using `PER_WORKER`.", DeprecationWarning)
    elif shuffle is False:
        shuffle = None
    else:
        raise ValueError(f"`shuffle={shuffle}` not recognized.")
    return shuffle


def _shuffle_df(df, size=None):
    """Shuffles a DataFrame, returning a new dataframe with randomly
    ordered rows"""
    size = size or len(df)
    # NOTE: We can use np.arange for both gpu and cpu-backed
    # dataframes once NEP-35 is fully accepted (`like` argument).
    # This should be available in numpy>=1.20
    if isinstance(df, pd.DataFrame):
        arr = np.arange(size)
        np.random.shuffle(arr)
    else:
        arr = cp.arange(size)
        # Note that np.random.shuffle "should" Work for both gpu
        # and cpu (via NEP-18), but it seems the cupy API is
        # still needed here for correct behavior.  (Probably related
        # to https://github.com/cupy/cupy/issues/2824)
        cp.random.shuffle(arr)
    return df.iloc[arr]
