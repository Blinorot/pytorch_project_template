import numpy as np
import torch
from tqdm.auto import tqdm

import datasets
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class HFDataset(BaseDataset):
    """
    A dataset loaded from HuggingFace.
    """

    def __init__(
        self, dataset_name, image_column="image", split="train", *args, **kwargs
    ):
        """
        Args:
            dataset_name (str): dataset repo_id in HuggingFace Hub.
            image_column (str): name of the image column.
            split (str): partition name.
        """

        index = datasets.load_dataset(dataset_name)[split]
        if image_column != "img":
            index = index.rename_column(image_column, "img")

        super().__init__(index, *args, **kwargs)
