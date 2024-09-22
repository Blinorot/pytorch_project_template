import json
import re
from pathlib import Path

import torchaudio
from tqdm import tqdm

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CommonVoiceDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        self._data_dir = ROOT_PATH / "dataset_common_voice"
        self._regex = re.compile("[^a-z ]")
        self._dataset = load_dataset(
            "mozilla-foundation/common_voice_11_0",
            "en",
            cache_dir=self._data_dir,
            split=split,
        )
        index = self._get_or_load_index(split)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, split):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = []
            for entry in tqdm(self._dataset):
                assert "path" in entry
                assert Path(
                    entry["path"]
                ).exists(), f"Path {entry['path']} doesn't exist"
                entry["path"] = str(Path(entry["path"]).absolute().resolve())
                entry["text"] = self._regex.sub("", entry.get("sentence", "").lower())
                t_info = torchaudio.info(entry["path"])
                entry["audio_len"] = t_info.num_frames / t_info.sample_rate
                index.append(
                    {
                        "path": entry["path"],
                        "text": entry["text"],
                        "audio_len": entry["audio_len"],
                    }
                )
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
