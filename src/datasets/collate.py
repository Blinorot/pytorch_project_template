import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    """
    result_batch = {}

    # Audio: (1, T) -> (B, 1, T_max)
    audios = [item["audio"].transpose(0, 1) for item in dataset_items]
    audios_padded = pad_sequence(audios, batch_first=True).transpose(1, 2)
    result_batch["audio"] = audios_padded

    result_batch["audio_length"] = torch.tensor(
        [item["audio"].shape[1] for item in dataset_items], dtype=torch.long
    )

    texts_encoded = [item["text_encoded"].squeeze(0) for item in dataset_items]
    texts_encoded_padded = pad_sequence(texts_encoded, batch_first=True)
    result_batch["text_encoded"] = texts_encoded_padded
    result_batch["text_encoded_length"] = torch.tensor(
        [item["text_encoded"].shape[1] for item in dataset_items], dtype=torch.long
    )

    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    return result_batch