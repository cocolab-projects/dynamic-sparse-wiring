import torch

from torch import nn


class RNNRouter(nn.Module):

    def __init__(self, hidden_size, operations, use_input_embedding=True,
                learn_inital_hidden=False, weight_tie_output_layer=True, bias=True) -> None:
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
            operations += 1
            self.first_operation = torch.zeros(1, operations).scatter_(1, torch.tensor([[0]]), 1)

        self.operation_embeddings = nn.Parameter(
            torch.empty(operations, hidden_size).uniform_(-0.1, 0.1))
        self.use_input_embedding = use_input_embedding
        self.rnn_cell = nn.GRUCell(hidden_size, hidden_size, bias=bias)
        self.output_layer = nn.Linear(hidden_size, operations, bias=bias)

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

        if last_decision is None:
            if self.use_input_embedding:
                last_decision = self.first_operation @ self.operation_embeddings
                # WARNING: might want to use repeat here
                last_decision = last_decision.expand(batch_size, -1)
            else:
                # Could be just zeros (?)
                last_decision = torch.zeros(batch_size, self.hidden_size).uniform_(-0.1, 0.1)
        else:
            last_decision = last_decision @ self.operation_embeddings

        hidden = self.rnn_cell(last_decision, hidden_state)
        return hidden, self.output_layer(hidden)
