import torch
from torch import Tensor, nn


class WhiteNoise(nn.Module):
    def __init__(self, min_snr_db=10, max_snr_db=30):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

    def forward(self, data: Tensor):
        """
        Args:
            data (Tensor): audio tensor, (1, T) or (B, 1, T)
        """
        if not self.training:
            return data

        snr_db = torch.empty(data.size(0)).uniform_(self.min_snr_db, self.max_snr_db)
        
        # Power of signal
        signal_power = data.pow(2).mean(dim=-1, keepdim=True)
        # Power of noise
        noise_power = signal_power / (10 ** (snr_db / 10)).to(data.device).unsqueeze(-1)
        
        noise = torch.randn_like(data) * noise_power.sqrt()
        return data + noise

