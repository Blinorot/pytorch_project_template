from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as T

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # Здесь считаем спектрограмму на GPU

        # Переименовываем результат из Batch Transform для модели
        if "audio" in batch and isinstance(batch["audio"], torch.Tensor):
            spec = batch["audio"]
            if spec.dim() == 4:
                spec = spec.squeeze(1)
            batch["spectrogram"] = spec
            
            if "audio_length" in batch:
                batch["spectrogram_length"] = (batch["audio_length"] // 200) + 1
                batch["spectrogram_length"] = batch["spectrogram_length"].to(self.device)
            else:
                batch["spectrogram_length"] = torch.full(
                    (spec.size(0),), spec.size(-1), device=spec.device, dtype=torch.long
                )

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                outputs = self.model(**batch)
                batch.update(outputs)

                all_losses = self.criterion(**batch)
                batch.update(all_losses)

            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            with torch.no_grad():
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    outputs = self.model(**batch)
                    batch.update(outputs)

                    all_losses = self.criterion(**batch)
                    batch.update(all_losses)

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            value = met(**batch)
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics.update(met.name, value)
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        # Берем первый элемент батча, отрезаем от графа и копируем на CPU
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        
        # Очищаем matplotlib после каждого использования, чтобы не копились фигуры в RAM
        import matplotlib.pyplot as plt
        image_tensor = plot_spectrogram(spectrogram_for_plot)
        
        # Переводим в формат, который точно поймет CometML/WandB (H, W, C)
        # или просто в PIL.Image
        image = T.ToPILImage()(image_tensor)
        
        self.writer.add_image("spectrogram", image)
        plt.close('all') # Гарантированно закрываем все фигуры

    def log_predictions(
        self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        log_probs_detached = log_probs.detach().cpu()
        argmax_inds = log_probs_detached.argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.detach().cpu().numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        # Beam search predictions
        probs = torch.exp(log_probs_detached)
        lengths = log_probs_length.detach().cpu().numpy()
        beam_texts = []
        for i in range(min(len(probs), examples_to_log)):
            beam_results = self.text_encoder.ctc_beam_search(
                probs[i][: lengths[i]], beam_size=10
            )
            beam_texts.append(beam_results[0][0])

        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path, beam_texts))

        rows = {}
        for pred, target, raw_pred, audio_path, beam_pred in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100
            beam_wer = calc_wer(target, beam_pred) * 100
            beam_cer = calc_cer(target, beam_pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "predictions": pred,
                "beam_predictions": beam_pred,
                "wer": wer,
                "cer": cer,
                "beam_wer": beam_wer,
                "beam_cer": beam_cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )