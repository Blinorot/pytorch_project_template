import shutil

import numpy as np
import safetensors
import safetensors.torch
import torchvision
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class CIFARDataset(BaseDataset):
    """
    CIFAR10 dataset

    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, name="train", *args, **kwargs):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "cifar" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "cifar" / name
        data_path.mkdir(exist_ok=True, parents=True)

        transform = torchvision.transforms.ToTensor()
        mnist_data = torchvision.datasets.CIFAR10(
            str(data_path), train=(name == "train"), download=True, transform=transform
        )

        print(f"Parsing CIFAR Dataset metadata for part {name}...")
        # wrapper over torchvision dataset to get individual objects
        # with some small changes in BaseDataset, torchvision dataset
        # can be used as is without this wrapper
        # but we use wrapper
        for i in tqdm(range(len(mnist_data))):
            # create dataset
            img, label = mnist_data[i]

            save_dict = {"tensor": img}
            save_path = data_path / f"{i:06}.safetensors"
            safetensors.torch.save_file(save_dict, save_path)

            # parse dataset metadata and append it to index
            index.append({"path": str(save_path), "label": label})

        shutil.rmtree(data_path / "cifar-10-batches-py")  # remove

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
