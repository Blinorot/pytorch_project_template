from typing import List

import torch


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    # example of collate_fn
    result_batch["data_object"] = torch.vstack(
        [elem["data_object"] for elem in dataset_items]
    )
    result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
