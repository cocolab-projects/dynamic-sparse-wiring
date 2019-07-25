from typing import Callable

import torch

from subset_operator import subset_operator

batch_size = 2 
beams = 2
operations = 5


def beam_search(step_function: Callable, beams: int, operations: int,
                preserve_zeros_grad=True):
    '''
    Args:
        step_function:
        beams:
        operations:
        preserve_zeros_grad:
    '''
    state = torch.zeros(batch_size, beams, operations)

    # TODO: placeholder
    # use top-k function to return these
    indices = torch.tensor([[2, 1], [2, 0]], dtype=torch.int64)
    values = torch.tensor([[0., 1., 1., 0., 0.],
                           [1., 0., 1., 0., 0.]],
                          dtype=torch.float32, requires_grad=True)

    # collecting all places where we selected an operation
    collected_values = torch.gather(dim=1, index=indices)
    # indexing over operations (the last dim)
    state = state.scatter(2, indices.unsqueeze(2), collected_values.unsqueeze(2))

    if preserve_zeros_grad:
        # Preserve the negative gradient for zeros, while respecting operations
        # selected in other beams.

        # As an example...
        #         v copy the gradient from these zeros
        # 0 1 1 0 0
        # 1 0 1 0 0
        #   ^ but don't copy this one as there is 1 in the column
        expanded_values = torch.unsqueeze(values, dim=1).expand(-1, beams, -1)
        mask = ~expanded_values.byte()
        state[mask] = expanded_values[mask]
    
    state
