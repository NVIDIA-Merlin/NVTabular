import torch

from nvtabular.framework_utils.torch import features


class SequentialBlock(torch.nn.Sequential):
    def __rrshift__(self, other):
        return right_shift_module(self, other)

    def __rshift__(self, other):
        return right_shift_module(other, self)


def right_shift_module(self, other):
    if isinstance(other, list):
        left_side = [features.FilterFeatures(other)]
    else:
        left_side = other.layers if isinstance(other, SequentialBlock) else [other]
    right_side = self.layers if isinstance(self, SequentialBlock) else [self]

    sequential = left_side + right_side
    return SequentialBlock(*sequential)
