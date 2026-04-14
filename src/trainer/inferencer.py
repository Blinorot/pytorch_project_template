import torch
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
        accelerator,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            accelerator (Accelerator): accelerator for handing processes and GPUs.
            device (str | torch.device): device for tensors/model. Must be the same as
                accelerator.device.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data. Evaluation will be done on all provided sets.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.accelerator = accelerator
        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = dataloaders

        # path definition
        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.accelerator.is_main_process:
            if self.metrics["inference"] is not None:
                self.evaluation_metrics = MetricTracker(
                    *[m.name for m in self.metrics["inference"]],
                    writer=None,
                )
            else:
                self.evaluation_metrics = None
        else:
            self.evaluation_metrics = None

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

    def process_batch(self, batch_idx, batch, part):
        """
        Run batch through the model and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk

        batch_size = batch["logits"].shape[0]
        current_id = batch_idx * batch_size

        rank = self.accelerator.process_index
        full_save_dir = self.save_path / part / f"rank_{rank}"
        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            logits = batch["logits"][i].detach().cpu().clone()
            label = batch["labels"][i].detach().cpu().clone()
            pred_label = logits.argmax(dim=-1)

            output_id = current_id + i

            output = {
                "pred_label": pred_label,
                "label": label,
            }

            if self.save_path is not None:
                # you can use safetensors or other lib here
                output_save_path = full_save_dir / f"output_{output_id}.pth"
                # prevent overwriting
                if not output_save_path.exists():
                    torch.save(output, output_save_path)

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

        if self.accelerator.is_main_process:
            if self.evaluation_metrics is not None:
                self.evaluation_metrics.reset()

        # create save dir for each rank to avoid overwrite
        rank = self.accelerator.process_index
        if self.save_path is not None:
            (self.save_path / part / f"rank_{rank}").mkdir(exist_ok=True, parents=True)

        self.accelerator.wait_for_everyone()

        with torch.no_grad():
            loader_iterator = enumerate(dataloader)

            if self.accelerator.is_main_process:
                loader_iterator = tqdm(
                    loader_iterator, desc=part, total=len(dataloader)
                )
            for batch_idx, batch in loader_iterator:
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                )
                # gathering must be called on all processes to avoid deadlock
                # self.evaluation_metrics is none for other processes
                if self.metrics["inference"] is not None:
                    gathered_batch = self._gather_batch_for_metrics(batch)
                    if self.accelerator.is_main_process:
                        self._update_metrics(
                            gathered_batch,
                            self.evaluation_metrics,
                            gather_loss=False,
                        )

        logs = {}
        if self.accelerator.is_main_process:
            if self.evaluation_metrics is not None:
                logs = self.evaluation_metrics.result()

        return logs
