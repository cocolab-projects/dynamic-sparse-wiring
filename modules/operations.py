'''
Provides some definitions
'''
from typing import Callable, List

import torch

from torch import nn
from dataclasses import dataclass


#pylint: disable=abstract-method
@dataclass
class Operations(nn.Module):
    '''
    Stores some metadata to avoid recomputing how many operations there are or
    how
    '''
    base_dictionary: dict
    operations: nn.ModuleList = None
    names: List[str] = None
    size: int = None

    #pylint: disable=unused-argument
    def __post_init__(self, *args) -> None:
        super().__init__()
        self.names = [name for name in self.base_dictionary.keys()]
        self.operations = nn.ModuleList(
            [module for module in self.base_dictionary.values()])
        self.size = len(self.modules)


class LambdaModule(nn.Module):
    '''
    Wraps a function to allow storage in a ModuleList.
    '''

    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.function = function

    # pylint: disable=arguments-differ
    def forward(self, input_: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.function(input_, *args, **kwargs)


def simple_activation_operations() -> Operations:
    '''
        Given 3 activation functions
    '''
    definition = {
        'sigmoid': LambdaModule(torch.sigmoid),
        'tanh': LambdaModule(torch.tanh),
        'relu': LambdaModule(torch.relu),
        'identity': nn.Identity(),
    }

    #pylint: disable-msg=too-many-function-args
    return Operations(definition)


def linear_layer_modules(hidden_size=128) -> Operations:
    '''
    # 3 linear layers and a identity
    '''
    hidden_shape = (hidden_size, hidden_size)

    definition = {
        'identity': nn.Identity(),
        'linear_0': nn.Linear(*hidden_shape),
        'linear_1': nn.Linear(*hidden_shape),
        'linear_2': nn.Linear(*hidden_shape),
    }

    #pylint: disable-msg=too-many-function-args
    return Operations(definition)
