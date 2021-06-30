import torch

from nvtabular.framework_utils.torch import features


class SequentialBlock(features.TabularMixin, torch.nn.Sequential):
    def __rrshift__(self, other):
        return right_shift_module(self, other)

    def __rshift__(self, other):
        return right_shift_module(other, self)


def right_shift_module(self, other):
    if isinstance(other, list):
        left_side = [features.FilterFeatures(other)]
    else:
        left_side = list(other) if isinstance(other, SequentialBlock) else [other]
    right_side = list(self) if isinstance(self, SequentialBlock) else [self]
    sequential = left_side + right_side

    return SequentialBlock(*sequential)
