import random
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.trainer import Trainer
from src.utils import ROOT_PATH, saving_init

warnings.filterwarnings("ignore", category=UserWarning)


def set_random_seed(seed):
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    set_random_seed(config.trainer.seed)

    logger = config.get_logger("train")
    save_dir = ROOT_PATH / "saved" / config.trainer.save_dir
    saving_init(save_dir, config)
    OmegaConf.save(config, save_dir / "config.yaml")

    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    # setup data_loader instances
    dataloaders = instantiate(config.dataloaders)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.scheduler, optimizer=optimizer)

    # len_epoch = number of iterations for iteration-based training
    # len_epoch = None or len(dataloader) for epoch-based training
    len_epoch = config.trainer.get("len_epoch")

    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        len_epoch=len_epoch,
    )

    trainer.train()


if __name__ == "__main__":
    main()
