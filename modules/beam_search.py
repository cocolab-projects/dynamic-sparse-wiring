'''
Implementation of a beam search with a top-k relaxation.
'''

from typing import Callable, Union, List
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from modules.subset_operator import subset_operator


# A RoutingFunction given a hidden state and a one-hot vector representing the last
# decision (potentially None) returns a new hidden state and a distribution over
# decisions.
# TODO: Add correct type hint
RoutingFunction = Callable


@dataclass
class BeamSearchResult:
    trajectories: torch.Tensor
    trejectory_scores: torch.Tensor
    score_values: torch.Tensor
    log_probs: List


def masking_topk(input_: torch.Tensor, k: int):
    '''
    A top-k operation that has the same behavior as subset_operator. Applies top-k
    along the second dimension.

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
        root_state: the inital hidden state for the routing function, shared by all
        children nodes
        routing_function: a Callable that returns a new hidden state and a
        distribution over decisions
        temperature: used to normalize the softmax used for estimating top-k
        differentiable: use a top-k relaxation

    Returns:
        A sparse solution
    '''
    batch_size = root_state.size(0)
    device = root_state.device

    # stores past decisions
    trajectories = torch.zeros(max_depth, batch_size, beams, logits_size).to(device)
    trajectory_scores = torch.zeros(batch_size, beams, 1).to(device)

    expand_on_beams = lambda tensor: tensor.expand(-1, beams, -1)

    # TODO: Does reshape handle the clone? Still worried about order... (Try reshape on
    # a range of ints?)
    hidden_state = expand_on_beams(root_state.unsqueeze(dim=1)
                                   ).reshape(batch_size * beams, -1)
    last_decision = None

    for depth in range(max_depth):

        # current_state (batch_size x beams x decisions)
        hidden_state, logits = routing_function(hidden_state, last_decision)
        log_probabilities = F.log_softmax(logits, dim=1)

        # Score tensor has a shape (batch_size x beams x logits) and holds the scores
        # for the current local decisions
        scores = (log_probabilities.view(batch_size, beams, logits_size) +
                  trajectory_scores.expand(-1, -1, logits_size)).view(
                      batch_size, beams * logits_size)
        scores_mask, scores_indices = subset_operator(scores, k=beams,
                                                      tau=temperature, hard=True)

        # Gather broadcasts (pytorch/pull/23479) remove repeat.
        gathers = torch.cat([
            scores_mask,
            log_probabilities.view(batch_size, beams * logits_size)
        ], dim=0).gather(1, scores_indices.repeat(2, 1))
        # TODO: !!! Potentially incorrect (batch_size?) !!!
        score_values, beam_scores = torch.split(gathers, batch_size)

        trajectory_scores = (
            trajectory_scores.squeeze() + beam_scores.squeeze()).unsqueeze(-1)

        # selection is done with a collapsed beam dimension, we therefore need to
        # correct the indices to account for our scatter being on a tensor with a
        # beam dimension
        scores_indices = torch.remainder(scores_indices, logits_size)

        selected_decisions = trajectories[depth]
        # create one-hot vectors for each decision
        selected_decisions = selected_decisions.scatter(
            2, scores_indices.unsqueeze(2), score_values.unsqueeze(2))

        if preserve_zeros_grad:
            # TODO: Add non-masking copy

            # Preserve negative gradient for zeros
            #         v copy gradient
            # 0 1 1 0 0
            # 1 0 1 0 0
            #   ^ don't copy gradient as 1 exists in column

            # values expanded along beams
            reshaped_mask = scores_mask.view_as(selected_decisions)
            summed_mask = reshaped_mask.sum(1, keepdim=True)
            # bitwise_not is a workaround, ~ is unsupported for bool tensors
            zeros_mask = expand_on_beams(summed_mask).bool().bitwise_not()
            selected_decisions[zeros_mask] = reshaped_mask[zeros_mask]

        last_decision = selected_decisions.view(batch_size * beams, logits_size)
        trajectories[depth] = selected_decisions

    trajectory_scores = trajectory_scores - torch.logsumexp(trajectory_scores,
                                                            dim=1, keepdim=True)

    return BeamSearchResult(trajectories, trajectory_scores, score_values)

