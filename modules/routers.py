from typing import Tuple, Callable

import torch

from torch import nn


class RandomRouter(nn.Module):
<<<<<<< HEAD

    def __init__(self, hidden_size, operations, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.operations = operations

    def forward(self, hidden_state, last_decision=None):
        batch_size = hidden_state.size(0)
        hidden_state = torch.empty(batch_size, self.hidden_size)
        return hidden_state, torch.randn((batch_size, self.operations)).cuda()
        # torch.nn.functional.one_hot(
        #     torch.randint(0, self.operations, (batch_size,)))


class RNNRouter(nn.Module):
=======
    '''
    '''
>>>>>>> 717fd319383a8f2f000933c5ac717946e9b0d389

    def __init__(self, hidden_size: int, operations: int, **kwargs) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.operations = operations

    def forward(self, hidden_state, last_decision: bool = None):
        batch_size = hidden_state.size(0)
        device = self.hidden_state.device
        # TODO: Why can't we just return hidden_state?
        hidden_state = torch.empty(batch_size, self.hidden_size).to(device)
        return hidden_state, torch.randn((batch_size, self.operations))


class RNNRouter(nn.Module):
    '''
    '''

    def __init__(self, hidden_size: int, operations: int,
                 use_input_embedding: bool = True,
                 weight_tie_output_layer: bool = True,
                 use_bias: bool = True) -> None:

        super().__init__()
        self.hidden_size = hidden_size
        self.use_input_embedding = use_input_embedding

        if use_input_embedding:
            self.input_embedding = nn.Parameter(
                torch.empty(1, hidden_size).uniform_(-0.1, 0.1))

        self.operation_embeddings = nn.Parameter(
            torch.empty(operations, hidden_size).uniform_(-0.1, 0.1))
        self.use_input_embedding = use_input_embedding
        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size, bias=use_bias)
        self.output_layer = nn.Linear(hidden_size, operations, bias=use_bias)

        if weight_tie_output_layer:
            self.output_layer.weight = self.operation_embeddings

    def forward(self, hidden_state,
                last_decision: bool = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            hidden_state: the hidden state of the RNN
            last_decision: a batch of one-hot vectors based on what decision was
            last made
        '''
        batch_size = hidden_state.size(0)

<<<<<<< HEAD
        if last_decision is None and self.use_input_embedding:
            input_state = self.input_embedding.expand(batch_size, -1).clone().detach()
        elif last_decision is None:
            # Could be just zeros (?)
            input_state = torch.empty(batch_size, self.hidden_size).uniform_(-0.1, 0.1)
=======
        if last_decision is None:
            if self.use_input_embedding:
                input_state = self.input_embedding.expand(batch_size, -1).clone().detach()
            else:
                input_state = torch.empty(batch_size,
                                          self.hidden_size).uniform(-0.1, 0.1)
>>>>>>> 717fd319383a8f2f000933c5ac717946e9b0d389
        else:
            input_state = last_decision @ self.operation_embeddings

        hidden_state = self.rnn_cell(input_state, hidden_state)

        return hidden_state, self.output_layer(hidden_state)

