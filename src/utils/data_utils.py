from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.datasets.collate import collate_fn


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def get_dataloaders(config):
    # transforms or augmentations init
    augmentations = instantiate(config.augmentations)
    # dataset partitions init
    datasets = instantiate(config.datasets, transforms=augmentations)

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        partition_dataloader = instantiate(
            config.dataloader,
            dataset=datasets[dataset_partition],
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders
