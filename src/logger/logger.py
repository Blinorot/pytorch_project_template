import logging
import logging.config
from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json


def setup_logging(save_dir, log_config=None, default_level=logging.INFO, append=False):
    """
    Setup logging configuration
    """
    if log_config is None:
        log_config = str(ROOT_PATH / "src" / "logger" / "logger_config.json")
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file is not found in {log_config}.")
        logging.basicConfig(level=default_level, filemode="a" if append else "w")
