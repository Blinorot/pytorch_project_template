import logging
import secrets
import shutil
import string

from omegaconf import OmegaConf

from src.logger.logger import setup_logging
from src.utils.io_utils import ROOT_PATH


# https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py
def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


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
        run_id = generate_id(length=config.writer.id_length)

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
