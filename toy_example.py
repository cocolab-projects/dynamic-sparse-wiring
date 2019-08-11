import torch

from torch import nn
from torch import optim

from modules import RNNRouter
from modules import beam_search


BATCH_SIZE = 64
HIDDEN_SIZE = 32
DECISIONS = 3
ENV_DEPTH = 5
BEAMS = 2

class SubsetSelector(nn.Module):

    def __init__(self, hidden_size, decisions, env_depth):
        super().__init__()
        self.root_state = torch.empty(1, hidden_size).uniform_(-0.1, 0.1)
        self.router = RNNRouter(hidden_size, decisions)
        self.env_depth = env_depth
        self.decisions = decisions

    def forward(self, env: torch.Tensor, beams: int = 100, temp: float = 20):
        batch_size = env.size(0)
        root_state = self.root_state.expand(batch_size, -1)
        result = beam_search(root_state, routing_function=self.router,
                             logits_size=self.decisions, beams=beams,
                             max_depth=self.env_depth, temperature=temp)
        return result

env = torch.arange(1, DECISIONS + 1, dtype=torch.float).unsqueeze(0).expand(
                    ENV_DEPTH, -1).clone()
batched_env = env.unsqueeze(0).expand(BATCH_SIZE, -1, -1).clone()


subset_selector = SubsetSelector(HIDDEN_SIZE, DECISIONS, ENV_DEPTH)

optimizer = optim.Adam(subset_selector.parameters())

optimal_loss = sum(batched_env.min(dim=2).values.flatten())


for epoch in range(100):
    optimizer.zero_grad()
    result = subset_selector(batched_env, beams=BEAMS)
    permuted_result = result.trajectories.permute(1, 2, 0, 3)
    batched_result = permuted_result.reshape(-1, ENV_DEPTH, DECISIONS)

    loss = torch.sum(env * batched_result)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        # True loss only counts the best beam
        mask = torch.nn.functional.one_hot(
            result.trejectory_scores.argmin(1).flatten()).flatten()
        masked = batched_result[mask.bool()]
        true_loss = torch.sum(env * masked)

        print(f'''
    Epoch: {epoch},
        Loss: {loss.item()},
        True Loss: {true_loss.item()},
        Optimal Loss: {optimal_loss}
        ''')
        print((env * batched_result[0:4]).detach().numpy())
