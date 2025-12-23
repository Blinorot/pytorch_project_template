import torch
import torchaudio
from torch import Tensor, nn


class PitchShift(nn.Module):
    def __init__(self, sample_rate=16000, n_steps_range=(-2, 2)):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_steps_range = n_steps_range

    def forward(self, data: Tensor):
        if not self.training:
            return data
        
        n_steps = torch.empty(1).uniform_(*self.n_steps_range).item()
        return torchaudio.functional.pitch_shift(data, self.sample_rate, n_steps)

