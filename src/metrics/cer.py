from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        log_probs_detached = log_probs.detach().cpu()
        predictions = torch.argmax(log_probs_detached, dim=-1).numpy()
        lengths = log_probs_length.detach().cpu().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers = []
        log_probs_detached = log_probs.detach().cpu()
        probs = torch.exp(log_probs_detached)
        lengths = log_probs_length.detach().cpu().numpy()
        for prob_vec, length, target_text in zip(probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            beam_results = self.text_encoder.ctc_beam_search(
                prob_vec[:length], self.beam_size
            )
            pred_text = beam_results[0][0]
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
