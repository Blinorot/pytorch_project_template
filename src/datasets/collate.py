import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["img"] = torch.cat(
        [elem["img"].unsqueeze(0) for elem in dataset_items], dim=0
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
