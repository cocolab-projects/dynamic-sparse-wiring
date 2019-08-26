from typing import NamedTuple, Union

import torch
from torch import distributions
from torch.nn import functional as F

from modules.lml import lml


EPSILON = torch.finfo(torch.float32).tiny


class ValuesIndices(NamedTuple):
    '''
    Return type for subset_operator
    '''
    values: torch.Tensor
    indices: torch.LongTensor


def gumble_subset_operator(scores, k: int, tau: float = 1.0,
                    hard: bool = True) -> ValuesIndices:
    '''
    An implementation of [Reparameterizable Subset Sampling via Continuous
    Relaxations](https://arxiv.org/abs/1901.10517) and a top-k relaxation from
    [Neural Nearest Neighbors Networks](https://arxiv.org/abs/1810.12575).

    Args:

    Returns
    '''
    m = distributions.gumbel.Gumbel(torch.zeros_like(scores, dtype=torch.float),
                                    torch.ones_like(scores, dtype=torch.float))
    g = m.sample()
    scores = scores + g

    k_hot = torch.zeros_like(scores)
    one_hot_approx = torch.zeros_like(scores)

    for _ in range(k):
        k_hot_mask = torch.max(1.0 - one_hot_approx,
                               torch.full_like(scores, EPSILON))
        scores = scores + torch.log(k_hot_mask)
        one_hot_approx = F.softmax(scores / tau, dim=1)
        k_hot = k_hot + one_hot_approx

    _values, indices = torch.topk(k_hot, k, dim=1)

    if hard:
        k_hot_hard = torch.zeros_like(k_hot)
        # switched scatter from index_add to support batching
        k_hot = k_hot_hard.scatter(1, indices, 1.) - k_hot.detach() + k_hot

    return ValuesIndices(k_hot, indices)


def lml_subset_operator(scores, k, *args, **kwargs):
    topk_simplex = lml(scores, k)
    _values, indices = torch.topk(scores, k, dim=1)
    k_hot_hard = torch.zeros_like(scores)
    k_hot = k_hot_hard.scatter(1, indices, 1.) - topk_simplex.detach() + topk_simplex
    return ValuesIndices(k_hot, indices)
