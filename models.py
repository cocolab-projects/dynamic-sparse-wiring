from typing import Callable, Tuple

import torch

from torch import nn

from subset_operator import subset_operator


class BeamSearch:

    def __init__(self, router_function: Callable, beams: int, operations: int,
                 maximum_depth: int, temp: float, preserve_zeros_grad: bool = True):
        '''
        Args:
            router_function:
            beams:
            operations:
            maximum_depth:
            temp:
            preserve_zeros_grad:
        '''
        self.router_function = router_function
        self.beams = beams
        self.operations = operations
        self.maximum_depth = maximum_depth
        self.temp = temp
        self.preserve_zeros_grad = preserve_zeros_grad
 
    def find_trajectories(self, batch: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        '''
        Given a batch, find the 

        Args:
            batch:
            temperature:
        Returns:
            A path
        '''
        # stories the history of decisions
        trajectories = torch.zeros(depth, batch_size, beams, operations)

        # stores the scores for each beam (unordered unfortunately)
        trajectory_scores = torch.zeros(batch_size, beams, 1)

        hidden_state, operation_distribution = self.router_function(batch, last_decision=None)
        scores = self.calculate_scores(trajectories[0], operation_distribution)

        for depth in range(1, self.maximum_depth):
            # (hidden_state)
            hidden_state, operation_distribution = self.router_function(
                                                    hidden_state, last_decision)
            scores = self.calculate_scores(trajectories[depth], operation_distribution)
            score_mask, score_indices = self.apply_topk(scores, temperature)
            collected_scores = torch.gather(dim=1, index=score_indices)
            # requires a reshape
            # state[depth] = (batch_size, beam_size, operations)

            # copy to each one of the beams
            state[depth] = state[depth].scatter(2, score_indices.unsqueeze(2),
                                                collected_scores.unsqueeze(2))
            if preserve_zeros_grad:
                # Preserve the negative gradient for zeros, while masking 
                # operations selected in other beams.

                # As an example...
                #         v copy the gradient from these zeros
                # 0 1 1 0 0
                # 1 0 1 0 0
                #   ^ but don't copy this one as there is 1 in the column
                expanded_values = torch.unsqueeze(values, dim=1).expand(-1, beams, -1)
                mask = ~expanded_values.byte()
                state[depth][mask] = expanded_values[mask]

            trajectory_scores += self.
            last_decision = self._reshape_to_batch_operations(state[depth])
            
            return trajectories, trajectories_scores

    def _reshape_to_batches_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Returns a reshaped tensor with shape (batch_size x total_operations)
        where total operations is the combined beams.
        '''
        return tensor.view(-1, self.beams * self.operations)
    
    def _reshape_to_batches_beams_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        '''
        Returns a reshaped tensor
        '''
        return tensor.view(-1, self.beams, self.operations)

    def calculate_scores(self, operation_distribution: torch.Tensor,
                         current_beams: torch.Tensor) -> torch.Tensor:
        '''
        Given our current beams, we want to calculate the scores
        with length beams x operations.
        '''
        # operation_distribution (batch_size, beams x operations)
        log_operation_distribution = F.log_softmax(operation_distribution)
        # current_beam (batch_size, beams, operations) →
        #              (batch_size, beams x operations)
        scores = self._reshape_to_batches_operations(current_beam) + log_operation_distribution
        return scores

    def apply_topk(self, scores, temperature):
        '''
        Given the scores over all beams, return the index of the beams
        TODO: how do we deal with order (?)

        Args:
            operation_distribution:
        '''
        # (batch_size x operations)
        all_paths = self._reshape_to_batches_operations(operation_distribution)
        subset_operator(all_paths, k=self.beams, tau=self.temperature)
        return
