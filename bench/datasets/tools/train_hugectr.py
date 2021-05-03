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

import hugectr

from nvtabular.ops import get_embedding_sizes


def train_hugectr(workflow, devices, out_path):
    # Gets embeddings and devices
    embeddings = list(get_embedding_sizes(workflow).values())
    embeddings = [emb[0] for emb in embeddings]
    devices = [[int(d)] for d in list(devices)[0::2]]
    # Set solver and model
    solver = hugectr.solver_parser_helper(
        vvgpu=[[0]],
        max_iter=10000,
        max_eval_batches=100,
        batchsize_eval=2720,
        batchsize=2720,
        display=1000,
        eval_interval=3200,
        snapshot=3200,
        i64_input_key=True,
        use_mixed_precision=False,
        repeat_dataset=True,
    )
    optimizer = hugectr.optimizer.CreateOptimizer(
        optimizer_type=hugectr.Optimizer_t.SGD, use_mixed_precision=False
    )
    model = hugectr.Model(solver, optimizer)
    model.add(
        hugectr.Input(
            data_reader_type=hugectr.DataReaderType_t.Parquet,
            source=out_path + "/output/train/_file_list.txt",
            eval_source=out_path + "/output/valid/_file_list.txt",
            check_type=hugectr.Check_t.Non,
            label_dim=1,
            label_name="label",
            dense_dim=13,
            dense_name="dense",
            slot_size_array=embeddings,
            data_reader_sparse_param_array=[
                hugectr.DataReaderSparseParam(hugectr.DataReaderSparse_t.Localized, 26, 1, 26)
            ],
            sparse_names=["data1"],
        )
    )
    model.add(
        hugectr.SparseEmbedding(
            embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,
            max_vocabulary_size_per_gpu=15500000,
            embedding_vec_size=128,
            combiner=0,
            sparse_embedding_name="sparse_embedding1",
            bottom_name="data1",
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["dense"],
            top_names=["fc1"],
            num_output=512,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu1"],
            top_names=["fc2"],
            num_output=256,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu2"],
            top_names=["fc3"],
            num_output=128,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc3"], top_names=["relu3"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.Interaction,
            bottom_names=["relu3", "sparse_embedding1"],
            top_names=["interaction1"],
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["interaction1"],
            top_names=["fc4"],
            num_output=1024,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc4"], top_names=["relu4"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu4"],
            top_names=["fc5"],
            num_output=1024,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc5"], top_names=["relu5"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu5"],
            top_names=["fc6"],
            num_output=512,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc6"], top_names=["relu6"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu6"],
            top_names=["fc7"],
            num_output=256,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc7"], top_names=["relu7"]
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.InnerProduct,
            bottom_names=["relu7"],
            top_names=["fc8"],
            num_output=1,
        )
    )
    model.add(
        hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
            bottom_names=["fc8", "label"],
            top_names=["loss"],
        )
    )
    # Run training
    model.compile()
    model.summary()
    model.fit()
