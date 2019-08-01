'''
Provides a simple implementation of a beam search with an optional top-k
relaxation. 
'''

from typing import Callable, Union

import torch

# A RoutingFunction takes a hidden_state as its first argument and one-hot
# vectors representing the last decision (None if no prior decision)
RoutingFunction = Callable[torch.Tensor, Union[torch.Tensor, None]]


def masking_topk(input_: torch.Tensor, k: int):
    '''
    A top-k operation that has behavior like subset_operator. Applies along
    the second dimension. 

    >>> masking_topk(torch.tensor([1., 3., 2.]), k=2)
    [0, 1, 1]
    '''
    _values, indices = input_.topk(dim=1, k=k)
    return torch.zeros_like(input_).scatter(1, indices, 1)


def search_step(trajectory_scores: torch.Tensor) -> torch.Tensor:
    '''
    Takes a traversal step based on the current beam state.
    '''
    pass


def beam_search(root_state: torch.Tensor, routing_function: RoutingFunction,
                logits: int, beams: int, max_depth: int, temperature: float = 1.,
                differentiable: bool = True):
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
    batches_logits_reshape = lambda tensor: tensor.view(-1 , beams * logits)
    batches_beams_logits_reshape = lambda tensor: tensor.view(-1, beams, logits)

    # stores past decisions
    trajectories = torch.zeros(max_depth, batch_size,)
    # score of each beam (unordered)
    trajectory_scores = torch.zeros(batch_size, beams, 1)

    hidden_state = root_state
