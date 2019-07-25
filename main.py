'''
Based on github.com/ermongroup/subsets
'''

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


def subset_operator(scores: torch.Tensor, k: int, tau: float = 1.0,
                    hard: bool = False) -> ValuesIndices:
    '''
    An implementation of 

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
        scores += torch.log(k_hot_mask)
        one_hot_approx = F.softmax(scores / tau, dim=1)
        k_hot += one_hot_approx

    _values, indices = torch.topk(k_hot, k, dim=1)

    if hard:
        k_hot_hard = torch.zeros_like(k_hot)
        # switched scatter from index_add_ to support batching
        k_hot = k_hot_hard.scatter(1, indices, 1.) - k_hot.detach() + k_hot

    return ValuesIndices(k_hot, indices)


class Router(nn.Module):

    def __init__(self):
        pass

    def forward(self, x):
        pass


class RandomRouter(nn.Module):

    def __init__(self):
        pass

    def forward(self):
        pass


class RNNRouter(nn.Module):
    '''
    '''

    def __init__(self, num_of_operations, hidden_size, cell_type):
        '''
        Args:
            num_of_operations
            hidden_size:
            cell_type:
        '''
        super().__init__()
        self.operation_embeddings = nn.Parameter(
            torch.empty(num_of_operations, hidden_size))

        self.rnn = nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size)

        self.reset_weights()
    
    def reset_weights(self):
        nn.init.uniform_(self.operation_embeddings)
        self.rnn

    def forward(self, x: torch.Tensor,
                last_decision: Union[None, torch.Tensor] = None):
        '''
        Args:
            x:
            last_decision:
        Returns:

        '''
        last_operation_embedding = last_decision @ self.operation_embeddings
        hidden_state = self.rnn()
        last_decision


def beam_search():
    pass


class Supernetwork:
    pass


if __name__ == '__main__':
    examples = F.softmax(torch.randn(2, 5, requires_grad=True), dim=0)
    print(subset_operator(examples, k=2, hard=True))
    import ipdb; ipdb.set_trace()
