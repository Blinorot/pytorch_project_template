import warnings
from datetime import timedelta

import hydra
import torch
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from huggingface_hub import create_repo
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoModel

from src.datasets.data_utils import get_dataloaders
from src.model import register_models
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    kwargs = [
        InitProcessGroupKwargs(timeout=timedelta(seconds=3600)),
        DistributedDataParallelKwargs(
            find_unused_parameters=config.trainer.find_unused_parameters
        ),
    ]
    accelerator = Accelerator(
        device_placement=False,  # we set to False for precise control of devices in batches
        cpu=device == "cpu",
        kwargs_handlers=kwargs,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,  # we do scheduler.step() ourselves
    )
    device = accelerator.device
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config, resolve=True)
    if accelerator.is_main_process:
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
        # If the multi-node setup does not share the filesystem,
        # saving must be called from the *local* main process.
        # In such a case, call setup_saving_and_logging in the local process.
        # However, you still have to set writer and logger to None for
        # non-main (including local main) processes as well after
        # setting the save directory.
    else:
        logger = None
        writer = None

    # setup data_loader instances
    # batch_transforms are manually moved/prepared with accelerator
    dataloaders, batch_transforms = get_dataloaders(config, accelerator, logger)

    # enable automodel and autoconfig
    register_models()
    # build model architecture, then print to console
    if config.trainer.from_pretrained is None:
        model_config = instantiate(config.model, _convert_="all")
        if accelerator.is_main_process:
            logger.info(model_config)
        model = AutoModel.from_config(model_config)
    else:
        if accelerator.is_main_process:
            logger.info(
                f"Loading model weights from: {config.trainer.from_pretrained} ..."
            )
        model = AutoModel.from_pretrained(config.trainer.from_pretrained)
    if accelerator.is_main_process:
        logger.info(model)
        model_arch = type(model).__name__
    model.to(device)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    # Prepare objects. Dataloaders are already prepared
    model, optimizer, lr_scheduler, loss_function = accelerator.prepare(
        model, optimizer, lr_scheduler, loss_function
    )

    # register everything except model, optimizer, and scheduler that need to be saved
    # accelerator.register_for_checkpointing(...)

    if accelerator.is_main_process and config.trainer.hf_push_to_hub:
        create_repo(
            repo_id=config.trainer.hf_repo_id,
            private=config.trainer.hf_repo_is_private,
            repo_type="model",
            exist_ok=True,
        )

    if accelerator.is_main_process:
        num_processes = accelerator.num_processes
        num_samples = len(dataloaders["train"].dataset)
        total_epoch_len = config.trainer.epoch_len
        grad_accum_steps = config.trainer.gradient_accumulation_steps
        if total_epoch_len is None:
            # epoch-based training
            total_epoch_len = len(dataloaders["train"]) // grad_accum_steps
        num_steps = total_epoch_len * config.trainer.n_epochs
        batch_size = min(num_samples, config.dataloader.train.batch_size)
        effective_batch_size = grad_accum_steps * num_processes * batch_size
        mixed_precision = accelerator.mixed_precision
        logger.info(
            (
                f"Starting Training of model: {model_arch}\n"
                f"    Num Training Samples: {num_samples}\n"
                f"    Num Processes: {num_processes}\n"
                f"    Total Number of Steps: {num_steps}\n"
                f"    Effective Batch Size: {effective_batch_size}\n"
                f"    Per-Process Batch Size: {batch_size}\n"
                f"    Gradient Accumulation Steps: {grad_accum_steps}\n"
                f"    Mixed Precision: {mixed_precision}\n"
            )
        )

    trainer = Trainer(
        accelerator=accelerator,
        device=device,
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()
    accelerator.end_training()


if __name__ == "__main__":
    main()
