'''
'''

import torch

from torch import nn
from torch.nn import functional as F

from modules.routers import RNNRouter
from modules.beam_search import beam_search


class Supernetwork(nn.Module):

    def __init__(self, beams: int, max_depth: int):
        super().__init__()
        self.beams = beams

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
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

    def forward(self, x):
        batch_size = x.size(0)
        inital_hidden_state = self.net(x).view(batch_size, -1)

        # save copy for network
        rnn_hidden_state = inital_hidden_state[:]
        path, _score = beam_search(rnn_hidden_state, self.router, logits_size=4,
                                   beams=self.beams, max_depth=3)
        # path = (depth, batch_size, beams, logits)
        hidden_state = inital_hidden_state
        hidden_state = torch.unsqueeze(
            hidden_state, dim=1).expand(-1, self.beams, -1).clone().view(
                self.beams * batch_size, self.logits_size)

        for operations in path:
            # operations (batch_size, beams, logits) -> (batch_size * beams, logits)
            operations = operations.view(batch_size * self.beams,
                                         self.logits_size)
            outputs = []
            for index, each in enumerate(operations):
                selected_operation_value = torch.max(each)
                output = self.module_list[torch.argmax(each)](hidden_state[index])
                output *= selected_operation_value
                outputs.append(F.relu(output))
            hidden_state = torch.cat(outputs, dim=1)

        return self.output_layer(hidden_state)


if __name__ == '__main__':
    supernetwork = Supernetwork(10, 3)
    supernetwork(torch.randn(1, 3, 32, 32))
