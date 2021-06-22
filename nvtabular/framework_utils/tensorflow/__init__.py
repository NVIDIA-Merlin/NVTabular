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

# flake8: noqa
from .feature_column_utils import make_feature_column_workflow
from .features import (FilterFeatures, ConcatFeatures, StackFeatures, TabularLayer, AsSparseLayer,
                       AsDenseLayer, ParseTokenizedText, AsTabular)
from .layers.inputs import EmbeddingsLayer, TransformersTextEmbedding, InputFeatures
from .blocks.base import right_shift_layer, Block, BlockWithHead
from .blocks.dlrm import DLRMBlock
from .heads import Head, Task
from . import tfrs
from tensorflow.keras.layers import Layer

Layer.__rrshift__ = right_shift_layer
