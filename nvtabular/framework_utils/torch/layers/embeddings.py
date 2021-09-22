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


class ConcatenatedEmbeddings(torch.nn.Module):
    """Map multiple categorical variables to concatenated embeddings.

    Args:
        embedding_table_shapes: A dictionary mapping column names to
            (cardinality, embedding_size) tuples.
        dropout: A float.
        sparse_columns: A list of sparse columns

    Inputs:
        x: An int64 Tensor with shape [batch_size, num_variables].

    Outputs:
        A Float Tensor with shape [batch_size, embedding_size_after_concat].
    """

    def __init__(self, embedding_table_shapes, dropout=0.0, sparse_columns=()):
        super().__init__()
        for col in sparse_columns:
            assert col in embedding_table_shapes, f"{col} is not in embedding_table_shapes"
        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.Embedding(cat_size, emb_size, sparse=(col in sparse_columns))
                for col, (cat_size, emb_size) in embedding_table_shapes.items()
            ]
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        if len(x.shape) <= 1:
            x = x.unsqueeze(0)
        x = [layer(x[:, i]) for i, layer in enumerate(self.embedding_layers)]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return x


class MultiHotEmbeddings(torch.nn.Module):
    """Map multiple categorical variables to concatenated embeddings.

    Args:
        embedding_dict_shapes: A dictionary mapping column names to
            (cardinality, embedding_size) tuples.
        dropout: A float.

    Inputs:
        x: A dictionary with multi-hot column name as keys and a tuple
           containing the column values and offsets as values.

    Outputs:
        A Float Tensor with shape [batch_size, embedding_size_after_concat].
    """

    def __init__(self, embedding_table_shapes, dropout=0.0, mode="sum"):
        super().__init__()
        self.embedding_names = list(embedding_table_shapes.keys())
        self.embedding_layers = torch.nn.ModuleList(
            [
                torch.nn.EmbeddingBag(*embedding_table_shapes[key], mode=mode)
                for key in self.embedding_names
            ]
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        embs = []
        for n, key in enumerate(self.embedding_names):
            values, offsets = x[key]
            values = torch.squeeze(values, -1)
            # for the case where only one value in values
            if len(values.shape) == 0:
                values = values.unsqueeze(0)
            embs.append(self.embedding_layers[n](values, offsets[:, 0]))
        x = torch.cat(embs, dim=1)
        x = self.dropout(x)
        return x
