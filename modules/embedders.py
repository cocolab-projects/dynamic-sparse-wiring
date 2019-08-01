import torch

from torch import nn
from torch.nn import functional as F


class SimpleEmbedder(nn.Module):
    '''
    Given an image, returns a fixed length embedding by applying a
    separable_conv and a linear layer.
    '''
    def __init__(self, input_channels, channels, embedding_size, kernel_size=7,
                 use_bias=False):
        super().__init__()

        # TODO: Support different kernal_sizes
        # I used a large kernel size to try to capture 'larger' features
        self.separable_conv = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=(kernel_size, 1),
                      padding=1, bias=use_bias),
            nn.Conv2d(channels, channels, kernel_size=(1, kernel_size),
                      padding=2, bias=use_bias),
        )
        self.linear = nn.Linear(channels * 32 * 32, embedding_size, bias=use_bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        '''
        Returns a fixed length (embedding_size) embedding.
        '''
        hidden = F.relu(self.separable_conv(image))
        return self.linear(torch.flatten(hidden, 1))
