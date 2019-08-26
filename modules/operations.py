from typing import Callable, List

import torch

from torch import nn


class LambdaModule(nn.Module):

    def __init__(self, function: Callable):
        self.function = function 
    
    def forward(self, input_: torch.Tensor, *args, **kwargs):
        return self.function(input_, *args, **kwargs)


def simple_activation_operations() -> List:
    '''
        3 activation functions
    '''
    return [
        LambdaModule(torch.sigmoid),
        LambdaModule(torch.tanh),
        LambdaModule(torch.relu),
        nn.Identity(),
    ]


def linear_layer_modules(hidden_size=128) -> List:
    # 3 linear layers and a identity
    hidden_shape = (hidden_size, hidden_size)
    return [
        nn.Identity(),
        nn.Linear(*hidden_shape),
        nn.Linear(*hidden_shape),
        nn.Linear(*hidden_shape),
    ]
