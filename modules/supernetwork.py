'''
'''
from typing import List

import torch

from torch import nn
from torch.nn import functional as F

from modules.routers import RNNRouter
from modules.beam_search import beam_search


class Supernetwork(nn.Module):

    def __init__(self, beams: int, max_depth: int):
        '''
        Args:
            beams
            max_depth:
        '''
        super().__init__()
        self.beams = beams

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )

        self.module_list = nn.ModuleList([
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Identity(),
        ])
        self.logits_size = len(self.module_list)
        self.router = RNNRouter(128, self.logits_size)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, x, tie_output_to_router: List[str] = ['pathwise']):
        batch_size = x.size(0)
        inital_hidden_state = self.net(x).view(batch_size, -1)

        # save copy for network
        rnn_hidden_state = inital_hidden_state[:]
        result = beam_search(rnn_hidden_state, self.router, logits_size=4,
                                   beams=self.beams, max_depth=3)
        # path = (depth, batch_size, beams, logits)
        hidden_state = inital_hidden_state
        hidden_state = torch.unsqueeze(
            hidden_state, dim=1).expand(-1, self.beams, -1).clone().view(
                self.beams * batch_size, -1)

        for operations in result.trajectories:
            # operations (batch_size, beams, logits) -> (batch_size * beams, logits)
            operations = operations.view(batch_size * self.beams,
                                         self.logits_size)
            outputs = []
            for index, each in enumerate(operations):
                output = self.module_list[torch.argmax(each)](hidden_state[index])
                if 'operationwise' in tie_output_to_router:
                    selected_operation_value = torch.max(each)
                    output *= selected_operation_value
                outputs.append(F.relu(output))
            hidden_state = torch.stack(outputs, dim=0)
        # returns batch_size * beams seems like this will need to change when we
        # only run a few of the paths
        predictions = self.output_layer(hidden_state)
        if 'pathwise' in tie_output_to_router:
            predictions *= torch.prod(result.score_values)

        return predictions


if __name__ == '__main__':
    supernetwork = Supernetwork(10, 3)
    supernetwork(torch.randn(1, 3, 32, 32))
