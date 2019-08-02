from typing import Tuple

import torch

from torch import nn


class RNNRouter(nn.Module):

    def __init__(self, hidden_size, operations, use_input_embedding=True,
                learn_inital_hidden=False, weight_tie_output_layer=True,
                use_bias=True) -> None:
        '''
        Args:
            hidden_size:
            operations:
            use_input_embedding:
            learn_inital_hidden:
            weight_tie_output_layer
            bias:
        Returns

        TODO: Add support for LSTMs
        '''
        super().__init__()

        self.hidden_size = hidden_size
        self.use_input_embedding = use_input_embedding
        if use_input_embedding:
            self.input_embedding = torch.empty(1, hidden_size).uniform_(-0.1, 0.1)

        self.operation_embeddings = nn.Parameter(
            torch.empty(operations, hidden_size).uniform_(-0.1, 0.1))
        self.use_input_embedding = use_input_embedding
        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size, bias=use_bias)
        self.output_layer = nn.Linear(hidden_size, operations, bias=use_bias)

        if weight_tie_output_layer:
            self.output_layer.weight = self.operation_embeddings

        self.init_weights()

    def init_weights(self):
        # TODO: What's the general idea with initalizing RNNs???
        pass

    def forward(self, hidden_state, last_decision=None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            hidden_state: the hidden_state of the RNN
            last_decision: a batch of one-hot based on what decision was made last

        Returns:
        '''
        batch_size = hidden_state.size(0)

        if last_decision is None and self.use_input_embedding:
            input_state = self.input_embedding.expand(batch_size, -1).clone()
        elif last_decision is None:
            # Could be just zeros (?)
            input_state = torch.empty(batch_size, self.hidden_size).uniform_(-0.1, 0.1)
        else:
            input_state = last_decision @ self.operation_embeddings

        hidden = self.rnn_cell(input_state, hidden_state)
        return hidden, self.output_layer(hidden)
