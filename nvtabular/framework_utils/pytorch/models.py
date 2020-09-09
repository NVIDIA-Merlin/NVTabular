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
import torch
from nvtabular.framework_utils.pytorch.layers import ConcatenatedEmbeddings

class Model(torch.nn.Module):
    
    def __init__(self, embedding_table_shapes, num_continuous,
                 emb_dropout, layer_hidden_dims, layer_dropout_rates, max_output=None):
        super().__init__()
        self.max_output = max_output
        self.initial_cat_layer = ConcatenatedEmbeddings(embedding_table_shapes, dropout=emb_dropout)
        self.initial_cont_layer = torch.nn.BatchNorm1d(num_continuous)
        
        embedding_size = sum(emb_size for _, emb_size in embedding_table_shapes.values())
        layer_input_sizes = [embedding_size + num_continuous] + layer_hidden_dims[:-1]
        layer_output_sizes = layer_hidden_dims
        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm1d(output_size),
                torch.nn.Dropout(dropout_rate)
            )
            for input_size, output_size, dropout_rate
            in zip(layer_input_sizes, layer_output_sizes, layer_dropout_rates)
        )
        
        self.output_layer = torch.nn.Linear(layer_output_sizes[-1], 1)
        
    def forward(self, x_cat, x_cont):
        x_cat = self.initial_cat_layer(x_cat)
        x_cont = self.initial_cont_layer(x_cont)
        x = torch.cat([x_cat, x_cont], 1)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        if self.max_output:
            x = self.max_output * torch.sigmoid(x)
        x = x.view(-1)
        return x

def rmspe_func(y_pred, y):
    "Return y_pred and y to non-log space and compute RMSPE"
    y_pred, y = torch.exp(y_pred) - 1, torch.exp(y) - 1
    pct_var = (y_pred - y) / y
    return (pct_var**2).mean().pow(0.5)
    
def process_epoch(dataloader, model, train=False, optimizer=None, loss_func=None,):
    model.train(mode=train)
    with torch.set_grad_enabled(train):
        y_list, y_pred_list = [], []
        for x_cat, x_cont, y in iter(dataloader):
            y_list.append(y.detach())
            y_pred = model(x_cat, x_cont)
            y_pred_list.append(y_pred.detach())
            loss = loss_func(y_pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    epoch_loss = loss_func(y_pred, y).item()
    epoch_rmspe = rmspe_func(y_pred, y).item()
    return epoch_loss, epoch_rmspe    

