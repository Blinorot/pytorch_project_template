import torch

from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, metric, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        All calculations of metrics are called only from the main process on
        gathered objects. All gathered objects are on CPU. If you want to compute
        metrics on GPU, pass accelerator.device as device and put tensors on device
        inside the __call__ method.

        Args:
            metric (Callable): function to calculate metrics.
        """
        super().__init__(*args, **kwargs)
        self.metric = metric

    def __call__(
        self, logits: list[torch.Tensor], labels: list[torch.Tensor], **kwargs
    ):
        """
        Metric calculation logic.

        Note that all inputs are gathered across all ranks.
        Each input is [input_rank_0, input_rank_1, ...]
        Metric calculation logic needs to account for gathering.

        If you want to use this metric for monitoring the best model,
        make sure it returns a python object (e.g. float) instead of a tensor.

        Args:
            logits (list[Tensor]): gathered model output predictions.
            labels (list[Tensor]): gathered ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        # if the tensors are of the same shape, we can easily concat them
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        classes = logits.argmax(dim=-1)
        return self.metric(classes, labels).item()  # ensure float
