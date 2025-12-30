import torch
from pathlib import Path
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics.
            part (str): name of the partition.
        Returns:
            batch (dict): batch with model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

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

        # на T4 не поддержан bf16 и мы грубо в fp16 заинферим (хотя может стоит в fp32, чтобы случайно не вылететь за границы) 
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        with torch.amp.autocast(dtype=dtype, device_type="cuda"):
            outputs = self.model(**batch)
            batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            # Save predictions as text files
            log_probs = batch["log_probs"].detach().cpu()
            log_probs_length = batch["log_probs_length"].detach().cpu().numpy()
            audio_paths = batch["audio_path"]

            # We can use beam search if specified in config, but argmax is faster
            argmax_inds = log_probs.argmax(-1).numpy()

            for i in range(len(audio_paths)):
                inds = argmax_inds[i][: int(log_probs_length[i])]
                pred_text = self.text_encoder.ctc_decode(inds)
                
                audio_name = Path(audio_paths[i]).stem
                with (self.save_path / part / f"{audio_name}.txt").open("w") as f:
                    f.write(pred_text)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
