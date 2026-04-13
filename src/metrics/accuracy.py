import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Accuracy metric
        """
        super().__init__(*args, **kwargs)

    def __call__(
        self, logits: list[torch.Tensor], labels: list[torch.Tensor], **kwargs
    ):
        """
        Metric calculation logic.

        Note that all inputs are gathered across all ranks.
        Gathered object is a list containing objects from each rank.
        A rank_i tensor of shape Bx(*OtherDims) is converted into a list of tensors
        of shape (*OtherDims). Then, all lists are concatenated:
        [
        obj_rank_0_batch_elem_0, obj_rank_0_batch_elem_1, ...,
        obj_rank_1_batch_elem_0, obj_rank_1_batch_elem_1, ...,
        ...
        ].
        The result is a list of length B*N_Processes.
        Metric calculation logic needs to account for gathering.

        If you want to use this metric for monitoring the best model,
        make sure it returns a python object (e.g. float) instead of a tensor.

        Args:
            logits (list[Tensor]): gathered model output predictions.
            labels (list[Tensor]): gathered ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        # if the tensors are of the same shape, we can easily stack them
        logits = torch.stack(logits, dim=0)
        labels = torch.stack(labels, dim=0)
        classes = logits.argmax(dim=-1)
        # .item converts to a JSON serializable value (float)
        return (classes == labels).mean(dtype=torch.float32).item()
