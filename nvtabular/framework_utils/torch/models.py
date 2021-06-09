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
import torch

from nvtabular.framework_utils.torch.layers import ConcatenatedEmbeddings, MultiHotEmbeddings


class Model(torch.nn.Module):
    """
    Generic Base Pytorch Model, that contains support for Categorical and Continous values.

    Parameters
    ----------
    embedding_tables_shapes: dict
        A dictionary representing the <column>: <max cardinality of column> for all
        categorical columns.
    num_continuous: int
        Number of continuous columns in data.
    emb_dropout: float, 0 - 1
        Sets the embedding dropout rate.
    layer_hidden_dims: list
        Hidden layer dimensions.
    layer_dropout_rates: list
        A list of the layer dropout rates expressed as floats, 0-1, for each layer
    max_output: float
        Signifies the max output.
    """

    def __init__(
        self,
        embedding_table_shapes,
        num_continuous,
        emb_dropout,
        layer_hidden_dims,
        layer_dropout_rates,
        max_output=None,
        bag_mode="sum",
    ):
        super().__init__()
        self.max_output = max_output

        if isinstance(embedding_table_shapes, tuple):
            cat_shapes, mh_shapes = embedding_table_shapes
        else:
            cat_shapes = embedding_table_shapes
            mh_shapes = None

        embedding_size = 0
        if cat_shapes:
            self.initial_cat_layer = ConcatenatedEmbeddings(cat_shapes, dropout=emb_dropout)
            embedding_size += sum(emb_size for _, emb_size in cat_shapes.values())
        if mh_shapes:
            self.mh_cat_layer = MultiHotEmbeddings(mh_shapes, dropout=emb_dropout, mode=bag_mode)
            embedding_size += sum(emb_size for _, emb_size in mh_shapes.values())

        self.initial_cont_layer = torch.nn.BatchNorm1d(num_continuous)

        layer_input_sizes = [embedding_size + num_continuous] + layer_hidden_dims[:-1]
        layer_output_sizes = layer_hidden_dims

        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(output_size),
                torch.nn.Dropout(dropout_rate),
            )
            for input_size, output_size, dropout_rate in zip(
                layer_input_sizes, layer_output_sizes, layer_dropout_rates
            )
        )

        self.output_layer = torch.nn.Linear(layer_output_sizes[-1], 1)

    def forward(self, x_cat, x_cont):
        concat_list = []

        if isinstance(x_cat, tuple):
            sh_cat, mh_cat = x_cat
        else:
            sh_cat = x_cat
            mh_cat = None

        if sh_cat is not None:
            sh_cat = self.initial_cat_layer(sh_cat)
            concat_list.append(sh_cat)

        if mh_cat is not None:
            mh_cat = self.mh_cat_layer(mh_cat)
            concat_list.append(mh_cat)

        if x_cont is not None:
            x_cont = self.initial_cont_layer(x_cont)
            concat_list.append(x_cont)

        x = torch.cat(concat_list, 1)

        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)

        if self.max_output:
            x = self.max_output * torch.sigmoid(x)

        return x.view(-1)
