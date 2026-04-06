import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch):
        """
        Run batch through the model, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if self.is_train:
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                batch.update(outputs)

                all_losses = self.criterion(**batch)
                batch.update(all_losses)

                # sum of all losses is always called loss
                grad_norm = None
                self.accelerator.backward(batch["loss"])
                if self.accelerator.sync_gradients:
                    grad_norm = self._get_grad_norm()
                    self._clip_grad_norm()
                    self.optimizer.step()
                    optimizer_step_was_skipped = (
                        self.accelerator.optimizer_step_was_skipped
                    )
                    if self.lr_scheduler is not None and not optimizer_step_was_skipped:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # control actual updates
                optimizer_did_step = self.accelerator.sync_gradients
        else:
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

            optimizer_did_step = None
            grad_norm = None

        return batch, optimizer_did_step, grad_norm

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
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
