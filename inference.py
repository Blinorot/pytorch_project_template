import warnings
from datetime import timedelta

import hydra
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from hydra.utils import instantiate
from transformers import AutoModel

from src.datasets.data_utils import get_dataloaders
from src.model import register_models
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    kwargs = [
        InitProcessGroupKwargs(timeout=timedelta(seconds=3600)),
    ]
    accelerator = Accelerator(
        device_placement=False,  # we set to False for precise control of devices in batches
        cpu=device == "cpu",
        kwargs_handlers=kwargs,
    )
    device = accelerator.device
    set_random_seed(config.inferencer.seed)

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, accelerator, None)

    # enable automodel and autoconfig
    register_models()
    # build model architecture, then print to console
    assert config.inferencer.from_pretrained is not None, "Provide model checkpoint."
    if accelerator.is_main_process:
        print(f"Loading model weights from: {config.inferencer.from_pretrained} ...")
    model = AutoModel.from_pretrained(config.inferencer.from_pretrained)
    model.to(device)

    if accelerator.is_main_process:
        print(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # Prepare objects. Dataloaders are already prepared
    model = accelerator.prepare(model)

    # save_path for model predictions
    # we assume all nodes share the filesystem
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    if accelerator.is_main_process:
        save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        accelerator=accelerator,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
    )

    logs = inferencer.run_inference()

    if accelerator.is_main_process:
        for part in logs.keys():
            for key, value in logs[part].items():
                full_key = part + "_" + key
                print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
