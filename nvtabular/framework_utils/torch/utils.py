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


class FastaiTransform:
    def __init__(self, dataloader):
        self.data = DictTransform(dataloader)

    def transform(self, batch):
        concats = []
        for columns in [self.data.cats, self.data.conts]:
            # cols = [v for k,v in batch[0] if k in columns and isinstance(v, torch.Tensor)]
            cols = []
            for k, v in batch[0].items():
                if k in columns and not isinstance(v, tuple) and isinstance(v, torch.Tensor):
                    cols.append(v)
            concats.append(torch.cat(cols, axis=1))
        return (
            concats[0].to("cuda", dtype=torch.long),
            concats[1],
            batch[1].to("cuda", dtype=torch.long),
        )


class DictTransform:
    def __init__(self, dataloader):
        self.cats = dataloader.cat_names
        self.conts = dataloader.cont_names
        self.labels = dataloader.label_names

    def transform(self, batch):
        batch, labels = batch
        cats = None
        conts = None
        # take a part the batch and put together into subsets
        if self.cats:
            cats = self.create_stack(batch, self.cats)
        if self.conts:
            conts, _ = self.create_stack(batch, self.conts)
        return cats, conts, labels

    def create_stack(self, batch, target_columns):
        columns = []
        mh_s = {}
        for column_name in target_columns:
            target = batch[column_name]
            if isinstance(target, torch.Tensor):
                if target.is_sparse:
                    mh_s[column_name] = target
                else:
                    columns.append(target)
            # if not a tensor, must be tuple
            else:
                # multihot column type, appending tuple representation
                mh_s[column_name] = target
        if columns:
            if len(columns) > 1:
                columns = torch.cat(columns, 1)
            else:
                columns = columns[0].unsqueeze(1)
        return columns, mh_s


def process_epoch(
    dataloader,
    model,
    train=False,
    optimizer=None,
    loss_func=torch.nn.MSELoss(),
    transform=DictTransform,
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
    if isinstance(dataloader, torch.utils.data.DataLoader):
        target = dataloader.dataset
    else:
        target = dataloader
    transform = transform(target).transform
    model.train(mode=train)
    idx = 0
    with torch.set_grad_enabled(train):
        y_list, y_pred_list = [], []
        for idx, batch in enumerate(iter(dataloader)):
            x_cat, x_cont, y = transform(batch)
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
