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


def process_epoch(
    dataloader,
    model,
    train=False,
    optimizer=None,
    loss_func=torch.nn.MSELoss(),
    transform=None,
    amp=True,
    device=None,
):
    """
    The controlling function that loads data supplied via a dataloader to a model. Can be redefined
    based on parameters.

    Parameters
    -----------
    dataloader : iterator
        Iterator that contains the dataset to be submitted to the model.
    model : torch.nn.Module
        Pytorch model to run data through.
    train : bool
        Indicate whether dataloader contains training set.
    optimizer : object
        Optimizer to run in conjunction with model.
    loss_func : function
        Loss function to use, default is MSELoss.
    """
    model.train(mode=train)
    with torch.set_grad_enabled(train):
        y_list, y_pred_list = [], []
        for idx, batch in enumerate(iter(dataloader)):
            if transform:
                x_cat, x_cont, y = transform(batch)
            else:
                x_cat, x_cont, y = batch
            if device:
                x_cat = x_cat.to(device)
                x_cont = x_cont.to(device)
                y = y.to(device)
            y_list.append(y.detach())
            # maybe autocast goes here?
            if amp:
                with torch.cuda.amp.autocast():
                    y_pred = model(x_cat, x_cont)
                    y_pred_list.append(y_pred.detach())
                    loss = loss_func(y_pred, y)
            else:
                y_pred = model(x_cat, x_cont)
                y_pred_list.append(y_pred.detach())
                loss = loss_func(y_pred, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    print(f"Total batches: {idx}")
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    epoch_loss = loss_func(y_pred, y).item()
    return epoch_loss, y_pred, y
