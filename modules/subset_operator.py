from typing import NamedTuple, Union

import torch

from torch import nn
from torch.nn import functional as F


EPSILON = torch.finfo(torch.float32).tiny


class ValuesIndices(NamedTuple):
    '''
    Return type for subset_operator
    '''
    values: torch.Tensor
    indices: torch.LongTensor

import torch
from torch.autograd import Function, Variable, grad
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sparsemax(Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape and reshape back after sparsemax
        original_size = input.size()
        input = input.view(-1, input.size(self.dim))

        dim = 1
        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.range(start=1, end=number_of_logits, device=device).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        output = self.output.view(original_size)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

sparsemax = Sparsemax(dim=1)

class LML(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, N):
        eps=1e-4
        n_iter=100
        branch=None
        if branch is None:
            if not x.is_cuda:
                branch = 10
            else:
                branch = 100
        branch = 1000

        single = x.ndimension() == 1
        orig_x = x
        if single:
            x = x.unsqueeze(0)
        assert x.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= N:
            y = (1.-1e-5)*torch.ones(n_batch, nx).type_as(x)
            if single:
                y = y.squeeze(0)
            ctx.save_for_backward(orig_x, y, torch.Tensor())
            return y + 1e-8

        x_sorted, _ = torch.sort(x, dim=1, descending=True)

        # The sigmoid saturates the interval [-7, 7]
        nu_lower = -x_sorted[:, N - 1] - 7.
        nu_upper = -x_sorted[:, N] + 7.

        ls = torch.linspace(0, 1, branch).type_as(x)

        for i in range(n_iter):
            r = nu_upper - nu_lower
            I = r > eps
            n_update = I.sum()
            if n_update == 0:
                break

            Ix = I

            nus = r[I].unsqueeze(1) * ls + nu_lower[I].unsqueeze(1)
            _xs = x[Ix].view(n_update, 1, nx) + nus.unsqueeze(2)
            fs = torch.sigmoid(_xs).sum(dim=2) - N
            # assert torch.all(fs[:,0] < 0) and torch.all(fs[:,-1] > 0)

            i_lower = ((fs < 0).sum(dim=1) - 1).long()
            J = i_lower < 0
            if J.sum() > 0:
                print('LML Warning: An example has all positive iterates.')
                i_lower[J] = 0

            i_upper = i_lower + 1

            nu_lower[I] = nus.gather(1, i_lower.unsqueeze(1)).squeeze()
            nu_upper[I] = nus.gather(1, i_upper.unsqueeze(1)).squeeze()

            if J.sum() > 0:
                nu_lower[J] -= 7.

        if np.any(I.cpu().numpy()):
            print('LML Warning: Did not converge.')

        nu = nu_lower + r / 2.
        y = torch.sigmoid(x + nu.unsqueeze(1))
        if single:
            y = y.squeeze(0)
        y = y + eps

        ctx.save_for_backward(orig_x, y, nu, torch.tensor([N]))

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # likely wrong :shrug:
        x, y, nu, N = ctx.saved_tensors
        N = N.item()

        single = x.ndimension() == 1

        if single:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)

        assert x.ndimension() == 2
        assert y.ndimension() == 2
        assert grad_output.ndimension() == 2

        n_batch, nx = x.shape
        if nx <= N:
            dx = torch.zeros_like(x)
            if single:
                dx = dx.squeeze()
            return dx, None

        Hinv = 1. / (1. / y + 1. / (1. - y))
        dnu = bdot(Hinv, grad_output) / Hinv.sum(dim=1)
        dx = - Hinv * (- grad_output + dnu.unsqueeze(1))

        if single:
            dx = dx.squeeze()

        return dx, None


lml = LML.apply


def subset_operator(scores: torch.Tensor, k: int, tau: float = 1.0,
                    hard: bool = False) -> ValuesIndices:
    '''
    An implementation of [Reparameterizable Subset Sampling via Continuous
    Relaxations](https://arxiv.org/abs/1901.10517) and a top-k relaxation from
    [Neural Nearest Neighbors Networks](https://arxiv.org/abs/1810.12575).

    Args:
        scores:
        k:
        tau:
        hard:
    Returns:
    '''
    m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores),
                                          torch.ones_like(scores))
    g = m.sample()
    scores = scores + g

    k_hot = torch.zeros_like(scores)
    one_hot_approx = torch.zeros_like(scores)

    for _ in range(k):
        k_hot_mask = torch.max(1.0 - one_hot_approx, torch.full_like(scores, EPSILON))
        scores = scores +  torch.log(k_hot_mask)
        one_hot_approx = F.softmax(scores / tau, dim=1)
        k_hot = k_hot + one_hot_approx

    _values, indices = torch.topk(k_hot, k, dim=1)

    if hard:
        k_hot_hard = torch.zeros_like(k_hot)
        # switched scatter from index_add_ to support batching
        k_hot = k_hot_hard.scatter(1, indices, 1.) - k_hot.detach() + k_hot

    return ValuesIndices(k_hot, indices)

def magic_subset_operator(scores, k, *args, **kwargs):
    topk_simplex = lml(scores, k)
    _values, indices = torch.topk(scores, k, dim=1)
    k_hot_hard = torch.zeros_like(scores)
    k_hot = k_hot_hard.scatter(1, indices, 1.) - topk_simplex.detach() + topk_simplex
    return ValuesIndices(k_hot, indices)