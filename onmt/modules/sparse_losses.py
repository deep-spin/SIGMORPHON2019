import torch
import torch.nn as nn
from torch.autograd import Function
from onmt.utils.misc import aeq as assert_equal

from onmt.modules.sparse_activations import sparsemax


def _fy_backward(ctx, grad_output):
    p_star, = ctx.saved_tensors
    grad = grad_output.unsqueeze(1) * p_star
    return grad


def _omega_sparsemax(p_star):
    return (1 - (p_star ** 2).sum(dim=1)) / 2


class SparsemaxLossFunction(Function):

    @classmethod
    def forward(cls, ctx, input, target):
        """
        input (FloatTensor): n x num_classes
        target (LongTensor): n, the indices of the target classes
        """
        assert_equal(input.shape[0], target.shape[0])

        p_star = sparsemax(input, 1)
        cls.p_star = p_star.clone().detach()
        loss = _omega_sparsemax(p_star)

        p_star.scatter_add_(1, target.unsqueeze(1),
                            torch.full_like(p_star, -1))
        loss += torch.einsum("ij,ij->i", p_star, input)

        ctx.save_for_backward(p_star)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        return _fy_backward(ctx, grad_output), None


sparsemax_loss = SparsemaxLossFunction.apply


class SparsemaxLoss(nn.Module):

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super(SparsemaxLoss, self).__init__()

    def forward(self, input, target):
        loss = sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss, SparsemaxLossFunction.p_star
