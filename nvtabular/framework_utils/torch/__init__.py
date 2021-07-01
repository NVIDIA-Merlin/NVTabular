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

from .features import FilterFeatures, StackFeatures, ConcatFeatures, AsTabular, TabularModule
from nvtabular.framework_utils.torch.models import Model
from nvtabular.framework_utils.torch.utils import process_epoch
from nvtabular.loader.torch import TorchAsyncItr
from nvtabular.framework_utils.torch.blocks.base import right_shift_module, SequentialBlock
from .layers.inputs import EmbeddingsModule, TableConfig, FeatureConfig, InputFeatures
from .blocks.mlp import MLPBlock

from torch.nn import Module


Module.__rrshift__ = right_shift_module

# __all__ = ["TorchAsyncItr", "Model", "process_epoch"]