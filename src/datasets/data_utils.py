from itertools import repeat

from hydra.utils import instantiate
from omegaconf import open_dict

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(skipped_dataloader, main_dataloader):
    """
    Wrapper function for endless dataloader.
    Used for the iteration-based training scheme.

    Can be used for the epoch-based training as well if the epoch_len
    is set to the length of the main dataloader.

    First yields from the skipped dataloader once. Then,
    infinitely yields from the main dataloader.

    If the skipped dataloader is None, yields directly from the main one.

    Args:
        skipped_dataloader (DataLoader | None): classic finite dataloader with
            the first few batches skipped because of training resuming.
        main_dataloader (DataLoader): classic finite dataloader.
    """
    if skipped_dataloader is not None:
        yield from skipped_dataloader
    for loader in repeat(main_dataloader):
        yield from loader


def prepare_batch_transforms(batch_transforms, accelerator):
    """
    Prepare batch_transforms with accelerate.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        accelerator (Accelerator): accelerator that controls processes.
    Returns:
        batch_transforms (dict[Callable] | None): processed transforms.
    """
    if batch_transforms is None:
        return None
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = accelerator.prepare(
                    transforms[transform_name].to(accelerator.device)
                )
    return batch_transforms


def get_dataloaders(config, accelerator, logger):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        accelerator (Accelerator): accelerator that controls processes.
        logger (Logger | None): logger that logs output.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    batch_transforms = prepare_batch_transforms(batch_transforms, accelerator)

    # dataset partitions init
    with accelerator.main_process_first():
        datasets = instantiate(
            config.datasets
        )  # instance transforms are defined inside

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        is_train = dataset_partition == "train"
        dataloader_config_type = "train" if is_train else "inference"
        dataloader_config = config.dataloader[dataloader_config_type].copy()
        with open_dict(dataloader_config):
            raise_error_for_small_datasets = dataloader_config.pop(
                "raise_error_for_small_datasets", False
            )

        if dataloader_config.batch_size > len(dataset):
            if raise_error_for_small_datasets:
                raise ValueError(
                    f"The batch size ({dataloader_config.batch_size}) cannot "
                    f"be larger than the dataset length ({len(dataset)})"
                )
            else:
                batch_size = len(dataset)
                if accelerator.is_main_process:
                    warning_text = (
                        f"Warning. Replacing config batch size {dataloader_config.batch_size} "
                        f"with {batch_size} due to the small size of {dataset_partition} "
                        "dataset. Set raise_error_for_small_datasets to true if you want "
                        "to raise an error in such a case instead."
                    )
                    if logger is not None:
                        logger.warning(warning_text)
                    else:
                        print(warning_text)
        else:
            batch_size = dataloader_config.batch_size

        partition_dataloader = instantiate(
            dataloader_config,
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            drop_last=is_train,
            shuffle=is_train,
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = accelerator.prepare(partition_dataloader)

    return dataloaders, batch_transforms
