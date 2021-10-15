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
from enum import Enum

class NVT_Dtypes(Enum):
    INT = "int",
    FLOAT = "float",

@dataclass(frozen=True)
class NVTDtype():
    dtype: str
    size: int = 0
    signed: bool = False
    is_list: bool = False


    def to(self, target_framework):
        type_dict[target_framework]._to(self)
        pass

    def create(self, dtype):
        if (isinstance(dtype, pd.))



type_dict = {
    pandas.dtype: PandasDtypes,
    numpy.dtype: NumpyDtypes,
    cudf.dtype: CudfDtypes,
}



class PandasDtypes():
    @classmethod
    def _to(cls, dtype):
        pass

    @classmethod
    def _from(cls, dtype):
        pass

class CudfDtypes():
    def _to(self):
        pass

    def _from(self):
        pass

class NumpyDtypes():
    def _to(self):
        pass

    def _from(self):
        pass
