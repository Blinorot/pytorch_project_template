import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    result_batch = {}

    # Spectrograms: (1, F, T) -> (B, F, T_max)
    # text_encoded: (1, L) -> (B, L_max)

    spectrograms = [item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items]
    spectrograms_padded = pad_sequence(spectrograms, batch_first=True).transpose(1, 2)
    result_batch["spectrogram"] = spectrograms_padded
    result_batch["spectrogram_length"] = torch.tensor(
        [item["spectrogram"].shape[2] for item in dataset_items], dtype=torch.long
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
