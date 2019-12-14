'''
Provides a simple implementation of a beam search with an optional top-k
relaxation.
'''

from typing import Callable, Union
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from modules.subset_operator import magic_subset_operator as subset_operator

# A RoutingFunction takes a hidden_state as its first argument and one-hot
# vectors representing the last decision (None if no prior decision)
RoutingFunction = Callable


@dataclass
class BeamSearchResult:
    trajectories: torch.Tensor
    trajectory_scores: torch.Tensor
    score_values: torch.Tensor


def masking_topk(input_: torch.Tensor, k: int):
    '''
    A top-k operation that has behavior like subset_operator. Applies along
    the second dimension.

    >>> masking_topk(torch.tensor([1., 3., 2.]), k=2)
    [0, 1, 1]
    '''
    _values, indices = input_.topk(dim=1, k=k)
    return torch.zeros_like(input_).scatter(1, indices, 1)


def beam_search(root_state: torch.Tensor, routing_function: RoutingFunction,
                logits_size: int, beams: int, max_depth: int,
                temperature: float = 1., preserve_zeros_grad: bool = True):
    '''
    Implements a beam search with a (optional) top-k relaxation based on
    [Reparameterizable Subset Sampling via Continuous Relaxations]
    (https://arxiv.org/abs/1901.10517) and [Neural Nearest Neighbors Networks]
    (https://arxiv.org/abs/1810.12575).

    Args:
        root_state: the inital hidden for the routing function, shared for all
            children nodes
        routing_function: a function that returns a new hidden_state along with
            a logits over decisions
        beams: the amount of paths to expand at each depth
        temperature: used to "soften" the a softmax estimating argmax similar to
            gumble-softmax
        differentiable: use a relaxed top-k operator

    Returns:
        A sparse solution
    '''
    batch_size = root_state.size(0)

    # stores past decisions
    trajectories = torch.zeros(max_depth, batch_size, beams, logits_size).cuda()
    # score of each beam (unordered)
    trajectory_scores = torch.zeros(batch_size, beams, 1).cuda()
    expand_on_beams = lambda tensor: tensor.expand(-1, beams, -1)

    hidden_state = expand_on_beams(root_state.unsqueeze(dim=1)).clone().reshape(
        batch_size * beams, -1)
    last_decision = None

    for depth in range(max_depth):
        # current_state (batch size x beams x decisions)
        hidden_state, logits = routing_function(hidden_state, last_decision)
        log_probabilities = F.log_softmax(logits, dim=1)
        # print(log_probabilities.exp())
        # import time
        # time.sleep(0.5)
        # The "score" matrix has a shape (batch size x beams x logits) and holds
        # all the scores we are locally selecting between
        scores = (log_probabilities.view(batch_size, beams, logits_size) +
                  trajectory_scores.expand(-1, -1, logits_size))
        scores = scores.view(batch_size, beams * logits_size)
        scores_mask, scores_indices = subset_operator(scores, k=beams,
                                                      tau=temperature,
                                                      hard=True)
        # once pytorch/pull/23479 (gather broadcasting) is merged - remove
        # cruff and gain speed improvement
        gathers = torch.cat([
            scores_mask,
            log_probabilities.view(batch_size, beams * logits_size)
        ], dim=0).gather(1, scores_indices.repeat(2, 1))
        score_values, beam_scores = torch.split(gathers, batch_size)

        trajectory_scores = (
            trajectory_scores.squeeze() + beam_scores.squeeze()).unsqueeze(-1)

        # selection is done with a collapsed dimension
        # modulo is used to correct indices
        scores_indices = torch.remainder(scores_indices, logits_size)
        # create a shallow copy of the current decision batch
        selected_decisions = trajectories[depth]
        # create one-hot vectors for each one of the decisions
        selected_decisions = selected_decisions.scatter(
            2, scores_indices.unsqueeze(2), score_values.unsqueeze(2))

        if preserve_zeros_grad:
            # Preserve negative gradient for zeros
            #         v copy gradient
            # 0 1 1 0 0
            # 1 0 1 0 0
            #   ^ don't copy as a 1 exists in the column

            # values expanded along the beams
            reshaped_mask = scores_mask.view_as(selected_decisions).bool()
            # summed_mask = reshaped_mask.sum(1, keepdim=True)
            # bitwise_not is a workaround, ~ is unsupported for bool tensors
            # zeros_mask = expand_on_beams(summed_mask).bool().bitwise_not()
            # selected_decisions[zeros_mask] = reshaped_mask[zeros_mask]
            selected_decisions[~reshaped_mask] = reshaped_mask[~reshaped_mask].float()

        last_decision = selected_decisions.view(batch_size * beams, logits_size)
        trajectories[depth] = selected_decisions

    trajectory_scores = trajectory_scores - torch.logsumexp(trajectory_scores,
                                                            dim=1, keepdim=True)
    return BeamSearchResult(trajectories, trajectory_scores, score_values)
