import abc
from typing import Union

import torch

from nvtabular.framework_utils.torch import features


class BlockMixin:
    def to_model(self, head, optimizer=torch.optim.Adam, **kwargs):
        return BlockWithHead(self, head, optimizer=optimizer, **kwargs)


class TabularBlock(features.TabularModule, BlockMixin):
    pass


class Block(BlockMixin, torch.nn.Module):
    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return self >> features.AsTabular(name)


class BuildableBlock(abc.ABC):
    @abc.abstractmethod
    def build(self, input_shape) -> Union[TabularBlock, Block]:
        raise NotImplementedError

    def __rrshift__(self, other):
        print("__rrshift__")

    def __rshift__(self, other):
        print("__rshift__")


class SequentialBlock(features.TabularMixin, torch.nn.Sequential):
    def __rrshift__(self, other):
        return right_shift_module(self, other)

    def __rshift__(self, other):
        return right_shift_module(other, self)

    def as_tabular(self, name=None):
        if not name:
            name = self.name

        return self >> features.AsTabular(name)


BlockType = Union[torch.nn.Module, Block]


class BlockWithHead(torch.nn.Module):
    def __init__(self, block: BlockType, head, optimizer=torch.optim.Adam, **kwargs):
        super().__init__()
        self.block = block
        self.head = head
        self.optimizer = optimizer

    def forward(self, inputs, *args, **kwargs):
        return self.head(self.block(inputs, *args, **kwargs), **kwargs)

    def compute_loss(self, inputs, targets, training: bool = False) -> torch.Tensor:
        logits = self(inputs, training=training)

        return self.head.compute_loss(targets, logits)

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
                loss = self.parent.compute_loss(*batch, training=True)
                self.log('train_loss', loss)

                return loss

            def configure_optimizers(self):
                optimizer = self.parent.optimizer(self.parameters(), lr=1e-3)
                return optimizer

        return BlockWithHeadLightning()


def right_shift_module(self, other):
    if isinstance(other, list):
        left_side = [features.FilterFeatures(other)]
    else:
        left_side = list(other) if isinstance(other, SequentialBlock) else [other]
    right_side = list(self) if isinstance(self, SequentialBlock) else [self]
    sequential = left_side + right_side

    return SequentialBlock(*sequential)
