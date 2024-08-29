import torch
from torch import nn


class RandomScale1D(nn.Module):
    """
    Randomly Scale 1D Input. Dummy Example of an augmentation.
    Used as an instance transform.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): scaled tensor.
        """
        scale = torch.randn(1)
        x = scale * x
        return x
