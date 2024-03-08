import torch

from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, metric, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = metric

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        classes = logits.argmax(dim=-1)
        return self.metric(classes, labels)
