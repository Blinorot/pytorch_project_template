import logging
import shutil

from omegaconf import OmegaConf

import wandb
from src.logger.logger import setup_logging
from src.utils.io_utils import ROOT_PATH


def resume_config(save_dir):
    saved_config = OmegaConf.load(save_dir / "config.yaml")
    run_id = saved_config.writer.run_id
    print(f"Resuming training from run {run_id}...")
    return run_id


def saving_init(save_dir, config):
    run_id = None

    if save_dir.exists():
        if config.trainer.get("resume_from") is not None:
            run_id = resume_config(save_dir)
        elif config.trainer.override:
            print(f"Overriding save directory '{save_dir}'...")
            shutil.rmtree(str(save_dir))
        elif not config.trainer.override:
            raise ValueError(
                "Save directory exists. Change the name or set override=True"
            )

    save_dir.mkdir(exist_ok=True, parents=True)

    if run_id is None:
        run_id = wandb.util.generate_id()

    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)

    OmegaConf.save(config, save_dir / "config.yaml")


def setup_saving_and_logging(config):
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    saving_init(save_dir, config)

    if config.trainer.get("resume_from") is not None:
        setup_logging(save_dir, append=True)
    else:
        setup_logging(save_dir, append=False)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    return logger
