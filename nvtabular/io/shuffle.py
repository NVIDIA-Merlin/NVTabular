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
from distutils.version import LooseVersion

import pandas as pd

_IGNORE_INDEX_SUPPORTED = pd.__version__ >= LooseVersion("1.3.0")


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


def _shuffle_df(df, size=None, keep_index=False):
    """Shuffles a DataFrame, returning a new dataframe with randomly
    ordered rows"""
    size = size or len(df)
    if isinstance(df, pd.DataFrame):
        if _IGNORE_INDEX_SUPPORTED:
            return df.sample(n=size, ignore_index=not keep_index)
        else:
            # Pandas<1.3.0
            if keep_index:
                return df.sample(n=size)
            return df.sample(n=size).reset_index(drop=True)
    else:
        return df.sample(n=size, keep_index=keep_index)
