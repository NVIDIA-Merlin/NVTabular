import abc
import inspect
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

from nvtabular.framework_utils.torch import features
from nvtabular.framework_utils.torch.heads import Head


class BlockMixin:
    def to_model(self, head, optimizer=torch.optim.Adam, block_output_size=None, **kwargs):
        if not block_output_size:
            if getattr(self, "input_size", None) and getattr(self, "forward_output_size", None):
                block_output_size = self.forward_output_size(self.input_size)
        if block_output_size:
            self.output_size = block_output_size

        out = BlockWithHead(self, head, optimizer=optimizer, **kwargs)

        if next(self.parameters()).is_cuda:
            out.to("cuda")

        return out


class TabularBlock(features.TabularModule, BlockMixin):
    pass


class Block(BlockMixin, torch.nn.Module):
    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return self >> features.AsTabular(name)


class SequentialBlock(features.TabularMixin, BlockMixin, torch.nn.Sequential):
    def __rrshift__(self, other):
        return right_shift_module(self, other)

    def __rshift__(self, other):
        return right_shift_module(other, self)

    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return self >> features.AsTabular(name)


class BuildableBlock(abc.ABC):
    @abc.abstractmethod
    def build(self, input_shape) -> Union[TabularBlock, Block, SequentialBlock]:
        raise NotImplementedError

    def __rrshift__(self, other):
        module = self.build(other.output_size())

        return right_shift_module(module, other)

    def __rshift__(self, other):
        print("__rshift__")


BlockType = Union[torch.nn.Module, Block]


class BlockWithHead(torch.nn.Module):
    def __init__(self, block: BlockType, head: Head, optimizer=torch.optim.Adam, **kwargs):
        super().__init__()
        self.block = block
        self.head = head
        self.optimizer = optimizer

    def forward(self, inputs, *args, **kwargs):
        return self.head(self.block(inputs, *args, **kwargs), **kwargs)

    def compute_loss(self, inputs, targets) -> torch.Tensor:
        block_outputs = self.block(inputs)
        return self.head.compute_loss(block_outputs, targets)

    def to_lightning(self):
        import pytorch_lightning as pl

        parent_self = self

        class BlockWithHeadLightning(pl.LightningModule):
            def __init__(self):
                super(BlockWithHeadLightning, self).__init__()
                self.parent = parent_self

            def forward(self, inputs, *args, **kwargs):
                return self.parent(inputs, *args, **kwargs)

            def training_step(self, batch, batch_idx):
                loss = self.parent.compute_loss(*batch)
                self.log('train_loss', loss)

                return loss

            def configure_optimizers(self):
                optimizer = self.parent.optimizer(self.parent.parameters(), lr=1e-3)

                return optimizer

        return BlockWithHeadLightning()

    def fit(self, dataloader, optimizer=torch.optim.Adam, num_epochs=1, amp=False, train=True, verbose=True):
        if isinstance(dataloader, torch.utils.data.DataLoader):
            dataset = dataloader.dataset
        else:
            dataset = dataloader

        if inspect.isclass(optimizer):
            optimizer = optimizer(self.parameters())

        self.train(mode=train)
        epoch_losses = []
        with torch.set_grad_enabled(mode=train):
            for epoch in range(num_epochs):
                losses = []
                batch_iterator = enumerate(iter(dataset))
                if verbose:
                    batch_iterator = tqdm(batch_iterator)
                for batch_idx, batch in batch_iterator:
                    x, y = batch
                    if amp:
                        with torch.cuda.amp.autocast():
                            loss = self.compute_loss(x, y)
                    else:
                        loss = self.compute_loss(x, y)

                    losses.append(float(loss))

                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                if verbose:
                    print(self.head.compute_metrics())
                epoch_losses.append(np.mean(losses))

        return np.array(epoch_losses)


def right_shift_module(self, other):
    if isinstance(other, list):
        left_side = [features.FilterFeatures(other)]
    else:
        left_side = list(other) if isinstance(other, SequentialBlock) else [other]
    right_side = list(self) if isinstance(self, SequentialBlock) else [self]
    sequential = left_side + right_side

    check_gpu = lambda x: next(x.parameters()).is_cuda
    need_moving_to_gpu = False
    if isinstance(self, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or check_gpu(self)
    if isinstance(other, torch.nn.Module):
        need_moving_to_gpu = need_moving_to_gpu or check_gpu(other)

    out = SequentialBlock(*sequential)

    if need_moving_to_gpu:
        out.to("cuda")

    return out
