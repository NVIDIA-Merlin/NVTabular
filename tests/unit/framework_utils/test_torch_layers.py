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
import pytest

# If pytorch isn't installed skip these tests. Note that the
# torch_dataloader import needs to happen after this line
torch = pytest.importorskip("torch")

# Must come after `pytest.importorskip("torch")`
from nvtabular.framework_utils.torch.layers.embeddings import ConcatenatedEmbeddings  # noqa: E402


def test_sparse_embedding_layer():
    embedding_table_shapes = {"col_a": (123, 4), "col_b": (45, 3)}

    emb_layer = ConcatenatedEmbeddings(embedding_table_shapes)
    assert not emb_layer.embedding_layers[0].sparse and not emb_layer.embedding_layers[1].sparse

    emb_layer = ConcatenatedEmbeddings(embedding_table_shapes, sparse_columns=["col_a"])
    assert emb_layer.embedding_layers[0].sparse and not emb_layer.embedding_layers[1].sparse
