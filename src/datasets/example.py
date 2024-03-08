import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ExampleDataset(BaseDataset):
    def __init__(
        self, input_length, n_classes, dataset_length, name="train", *args, **kwargs
    ):
        index_path = ROOT_PATH / "data" / "example" / name / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(input_length, n_classes, dataset_length, name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, input_length, n_classes, dataset_length, name):
        index = []
        data_path = ROOT_PATH / "data" / "example" / name
        data_path.mkdir(exist_ok=True, parents=True)

        number_of_zeros = int(np.log10(dataset_length)) + 1

        print("Creating Example Dataset")
        for i in tqdm(range(dataset_length)):
            example_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            example_data = torch.randn(input_length)
            example_label = torch.randint(n_classes, size=(1,)).item()
            torch.save(example_data, example_path)

            index.append({"path": str(example_path), "label": example_label})

        write_json(index, str(data_path / "index.json"))

        return index
