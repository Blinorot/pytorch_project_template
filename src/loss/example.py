import torch
from torch import nn


class ExampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        return {"loss": self.loss(logits, labels)}
