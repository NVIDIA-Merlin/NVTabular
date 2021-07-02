import abc
from typing import Union

import torch

from nvtabular.framework_utils.torch import features
from nvtabular.framework_utils.torch.heads import Head


class BlockMixin:
    def to_model(self, head: Head, optimizer=torch.optim.Adam, block_output_size=None, **kwargs):
        from nvtabular.framework_utils.torch.blocks.with_head import BlockWithHead

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
