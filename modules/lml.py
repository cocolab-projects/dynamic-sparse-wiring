#!/usr/bin/env python3
#
# Copyright 2019 Intel AI, CMU, Bosch AI

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from torch.autograd import Function, Variable, grad
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr


def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze()


class LML(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, N, eps, n_iter, branch):

        if branch is None:
            if not x.is_cuda:
                branch = 10
            else:
                branch = 100

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
            # adding a epsilon here helps training when some choices get close
            # to zero
            return y + eps

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
        # we also add the epsilon here
        y = y + eps

        ctx.save_for_backward(orig_x, y, nu)
        ctx.N = N

        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y, nu = ctx.saved_tensors
        N = ctx.N

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


def lml(input_, N, eps=1e-4, n_iter=100, branch=None):
    return LML.apply(input_, N, eps, n_iter, branch)


if __name__ == '__main__':
    input_ = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], requires_grad=True)
    input_.register_hook(lambda grad: print(grad))
    output = lml(input_, 3)
    print(output)
    torch.prod(output).backward()
