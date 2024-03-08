from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def get_dataloaders(config):
    augmentations = instantiate(
        config.augmentations
    )  # transforms or augmentations init
    datasets = instantiate(config.datasets, transforms=augmentations)

    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        partition_dataloader = DataLoader(
            datasets[dataset_partition],
            batch_size=config.trainer.batch_size,
            num_workers=config.trainer.num_workers,
            collate_fn=collate_fn,
            pin_memory=config.trainer.get("pin_memory", True),
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders
