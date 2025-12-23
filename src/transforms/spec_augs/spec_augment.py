import torchaudio
from torch import nn


class TimeMasking(nn.Module):
    def __init__(self, time_mask_param=30, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param, **kwargs)

    def forward(self, data):
        return self._aug(data)


class FrequencyMasking(nn.Module):
    def __init__(self, freq_mask_param=15, **kwargs):
        super().__init__()
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param, **kwargs)

    def forward(self, data):
        return self._aug(data)

