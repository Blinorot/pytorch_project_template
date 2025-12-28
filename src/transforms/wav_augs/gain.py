import torch
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, min_gain_in_db=-10, max_gain_in_db=10, p=0.5):
        super().__init__()
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db
        self.p = p

    def forward(self, data: Tensor):
        """
        Args:
            data (Tensor): audio tensor, (1, T) or (B, 1, T)
        """
        if not self.training or torch.rand(1) > self.p:
            return data

        gain_db = torch.empty(data.size(0), device=data.device).uniform_(
            self.min_gain_in_db, self.max_gain_in_db
        )

        gain = 10 ** (gain_db / 20)

        return data * gain.view(-1, 1, 1) if data.dim() == 3 else data * gain
