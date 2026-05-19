import os
import shutil
from abc import abstractmethod

import torch
from huggingface_hub import snapshot_download, upload_folder
from numpy import inf
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        accelerator,
        device,
        dataloaders,
        logger,
        writer,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        """
        Args:
            model (nn.Module): PyTorch model.
            criterion (nn.Module): loss function for model training.
            metrics (dict): dict with the definition of metrics for training
                (metrics[train]) and inference (metrics[inference]). Each
                metric is an instance of src.metrics.BaseMetric.
            optimizer (Optimizer): optimizer for the model.
            lr_scheduler (LRScheduler): learning rate scheduler for the
                optimizer.
            config (DictConfig): experiment config containing training config.
            accelerator (Accelerator): accelerator for handing processes and GPUs.
            device (str | torch.device): device for tensors/model. Must be the same as
                accelerator.device.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            logger (Logger): logger that logs output.
            writer (WandBWriter | CometMLWriter): experiment tracker.
            epoch_len (int | None): number of steps in each epoch for
                iteration-based training. If None, use epoch-based
                training (len(dataloader)).
            skip_oom (bool): skip batches with the OutOfMemory error.
            batch_transforms (dict[Callable] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
        """
        self.is_train = True

        self.config = config
        self.cfg_trainer = self.config.trainer

        self.accelerator = accelerator
        self.device = device
        self.skip_oom = skip_oom

        self.logger = logger
        self.log_step = config.trainer.get("log_step", 50)

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.train_dataloader = dataloaders["train"]
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        # set after resume to properly restore train dataloader state
        if epoch_len is None:
            # epoch-based training
            grad_accum_steps = self.config.trainer.gradient_accumulation_steps
            self.epoch_len = len(self.train_dataloader) // grad_accum_steps
        else:
            # iteration-based training
            self.epoch_len = epoch_len

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs
        self.global_step = -1  # total number of steps across all epochs
        self.epoch_step = -1  # number of steps in the current epoch
        self.seen_dataloader_batches = 0
        self.train_dataloader_length = len(self.train_dataloader)

        # configuration to monitor model performance and save best
        self.save_period = (
            self.cfg_trainer.save_period
        )  # checkpoint each save_period epochs
        self.monitor = self.cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = writer

        # define metrics
        self.metrics = metrics
        if self.accelerator.is_main_process:
            self.train_metrics = MetricTracker(
                *self.config.writer.loss_names,
                "grad_norm",
                *[m.name for m in self.metrics["train"]],
                writer=self.writer,
            )
            self.evaluation_metrics = MetricTracker(
                *self.config.writer.loss_names,
                *[m.name for m in self.metrics["inference"]],
                writer=self.writer,
            )
        else:
            self.train_metrics = None
            self.evaluation_metrics = None

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        skipped_dataloader = None
        if config.trainer.get("resume_from") is not None:
            if config.trainer.resume_from == "huggingface":
                resume_path = "huggingface"
            else:
                resume_path = self.checkpoint_dir / config.trainer.resume_from
            skipped_dataloader = self._resume_checkpoint(resume_path)

        # set after resume to properly restore the train dataloader state
        self.train_dataloader = inf_loop(
            skipped_dataloader=skipped_dataloader,  # first yield from this once
            main_dataloader=self.train_dataloader,  # then from this in inf loop
        )
        # note that we can do inf_loop even for the epoch-based training,
        # as the epoch length is controlled by the epoch_len always,
        # which makes inf_loop equivalent to re-iterating over
        # the dataloader each epoch

    def train(self):
        """
        Wrapper around training process to save model on keyboard interrupt.
        """
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            if self.accelerator.is_main_process:
                self.logger.info("Saving model on keyboard interrupt")
                # saving is done in the main process
                self._save_checkpoint(save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic:

        Training model for an epoch, evaluating it on non-train partitions,
        and monitoring the performance improvement (for early stopping
        and saving the best checkpoint).
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)
            self.epoch_step = -1

            # save logged information into logs dict
            logs = {"epoch": epoch}
            logs.update(result)

            if self.accelerator.is_main_process:
                # print logged information to the screen
                for key, value in logs.items():
                    self.logger.info(f"    {key:15s}: {value}")

                # evaluate model performance according to configured metric,
                # save best checkpoint as model_best
                best, stop_process, not_improved_count = self._monitor_performance(
                    logs, not_improved_count
                )

                if epoch % self.save_period == 0 or best:
                    self._save_checkpoint(save_best=best, only_best=True)

                if stop_process:  # early_stop
                    self.accelerator.set_trigger()

            # wait for the main process to finish logs and saving
            self.accelerator.wait_for_everyone()

            if self.accelerator.check_trigger():  # check early_stop trigger
                break

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch, including logging and evaluation on
        non-train partitions.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self.is_train = True
        self.model.train()
        if self.accelerator.is_main_process:
            self.train_metrics.reset()
            # +1 to log at the border
            self.writer.set_step(self.global_step + 1)
            self.writer.add_scalar("epoch", epoch)
            last_train_metrics = {}

        if self.accelerator.is_main_process:
            pbar = tqdm(total=self.epoch_len, desc="train")
            # progress bar resume
            pbar.update(self.epoch_step)

        for batch in self.train_dataloader:
            try:
                batch, did_optimizer_step, grad_norm = self.process_batch(
                    batch,
                )
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    if self.accelerator.is_main_process:
                        self.logger.warning("OOM on batch. Skipping batch.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # free some memory
                    continue
                else:
                    raise e

            # control the inner dataloader state
            self.seen_dataloader_batches = self.seen_dataloader_batches + 1
            self.seen_dataloader_batches = (
                self.seen_dataloader_batches % self.train_dataloader_length
            )

            if did_optimizer_step:
                self.global_step += 1
                self.epoch_step += 1

                if self.accelerator.is_main_process:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "last_loss": batch["loss"].item(),
                            "grad_norm": grad_norm,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "global_step": self.global_step,
                        }
                    )

                # log current results
                if self.epoch_step % self.log_step == 0:
                    # gathering must be called on all processes to avoid deadlock
                    gathered_batch = self._gather_batch_for_metrics(batch)

                    if self.accelerator.is_main_process:
                        # compute metrics
                        gathered_loss = self._update_metrics(
                            gathered_batch, self.train_metrics
                        )
                        self.train_metrics.update("grad_norm", grad_norm)

                        self.writer.set_step(self.global_step)
                        self.logger.debug(
                            "Train Epoch: {} {} Loss: {:.6f}".format(
                                epoch,
                                self._progress(self.epoch_step),
                                gathered_loss["loss"],
                            )
                        )
                        self.writer.add_scalar(
                            "learning rate", self.lr_scheduler.get_last_lr()[0]
                        )
                        self._log_scalars(self.train_metrics)
                        self._log_batch(self.epoch_step, batch)
                        # we don't want to reset train metrics at the start of every epoch
                        # because we are interested in recent train metrics
                        last_train_metrics = self.train_metrics.result()
                        self.train_metrics.reset()
            if self.epoch_step + 1 >= self.epoch_len:
                break

        logs = {}
        if self.accelerator.is_main_process:
            logs = last_train_metrics
            pbar.close()

        # Run val/test
        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            if self.accelerator.is_main_process:
                logs.update(
                    **{f"{part}_{name}": value for name, value in val_logs.items()}
                )

        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        if self.accelerator.is_main_process:
            self.evaluation_metrics.reset()
        with torch.no_grad():
            loader_iterator = enumerate(dataloader)

            if self.accelerator.is_main_process:
                loader_iterator = tqdm(
                    loader_iterator, desc=part, total=len(dataloader)
                )
            for batch_idx, batch in loader_iterator:
                batch, _, _ = self.process_batch(batch)
                # gathering must be called on all processes to avoid deadlock
                gathered_batch = self._gather_batch_for_metrics(batch)
                if self.accelerator.is_main_process:
                    self._update_metrics(gathered_batch, self.evaluation_metrics)
            if self.accelerator.is_main_process:
                # + 1 to log on the border
                self.writer.set_step(self.global_step + 1, part)
                self._log_scalars(self.evaluation_metrics)
                self._log_batch(
                    batch_idx, batch, part
                )  # log only the last batch during inference

        logs = {}
        if self.accelerator.is_main_process:
            logs = self.evaluation_metrics.result()
        return logs

    def _update_metrics(self, gathered_batch, metrics, gather_loss=True):
        """
        Update metrics using the current batch.

        Args:
            gathered_batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses. Must be the output of
                self._gather_batch_for_metrics.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
            gather_loss (bool): if False, skip updating loss metrics.
        Returns:
            gathered_loss (float): overall loss. Used for logging.
        """
        metrics_type = "train" if self.is_train else "inference"
        metric_funcs = self.metrics[metrics_type]

        # update metrics for each loss (in case of multiple losses)
        if gather_loss:
            gathered_loss = {}
            for loss_name in self.config.writer.loss_names:
                gathered_loss[loss_name] = gathered_batch[loss_name]  # list
                gathered_loss[loss_name] = (
                    torch.tensor([elem.item() for elem in gathered_loss[loss_name]])
                    .mean()
                    .item()
                )
                metrics.update(loss_name, gathered_loss[loss_name])
        else:
            gathered_loss = None

        for met in metric_funcs:
            metrics.update(met.name, met(**gathered_batch))

        return gathered_loss

    def _gather_batch_for_metrics(self, batch):
        """
        Gather tensor values across processes for metric computation.
        Non-tensor objects are left as-is.

        Gathered object is a list containing objects from each rank.
        A rank_i tensor of shape Bx(*OtherDims) is converted into a list of tensors
        of shape (*OtherDims). Then, all lists are concatenated:
        [
        obj_rank_0_batch_elem_0, obj_rank_0_batch_elem_1, ...,
        obj_rank_1_batch_elem_0, obj_rank_1_batch_elem_1, ...,
        ...
        ].
        The result is a list of length B*N_Processes.
        This is safer in case of varying tensor shapes between ranks,
        e.g. due to padding.
        Scalar tensors are reshaped to be a (1,) tensor.

        Metrics calculation need to account for this design.

        Args:
            batch (dict): current batch.
        Returns:
            gathered (dict): gathered batch.
        """
        gathered = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                if value.ndim == 0:
                    value = value.unsqueeze(-1)  # add dim for scalar tensors
                gathered_value = self.accelerator.gather_for_metrics(
                    value.detach().cpu(), use_gather_object=True
                )
                if self.accelerator.num_processes == 1:
                    # gather returns a tensor instead of list
                    # converting to list for consistency
                    gathered_value = list(gathered_value.unbind(dim=0))
                gathered[key] = gathered_value
            else:
                gathered[key] = value
        return gathered

    def _monitor_performance(self, logs, not_improved_count):
        """
        Check if there is an improvement in the metrics. Used for early
        stopping and saving the best checkpoint.

        Args:
            logs (dict): logs after training and evaluating the model for
                an epoch.
            not_improved_count (int): the current number of epochs without
                improvement.
        Returns:
            best (bool): if True, the monitored metric has improved.
            stop_process (bool): if True, stop the process (early stopping).
                The metric did not improve for too much epochs.
            not_improved_count (int): updated number of epochs without
                improvement.
        """
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                if self.accelerator.is_main_process:
                    self.logger.warning(
                        f"Warning: Metric '{self.mnt_metric}' is not found. "
                        "Model performance monitoring is disabled."
                    )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count >= self.early_stop:
                if self.accelerator.is_main_process:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        """
        Transforms elements in batch. Like instance transform inside the
        BaseDataset class, but for the whole batch. Improves pipeline speed,
        especially if used with a GPU.

        Each tensor in a batch undergoes its own transform defined by the key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform).
        """
        if self.batch_transforms is None:
            return batch
        # do batch transforms on device
        transform_type = "train" if self.is_train else "inference"
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](
                    batch[transform_name]
                )
        return batch

    def _clip_grad_norm(self):
        """
        Clips the gradient norm by the value defined in
        config.trainer.max_grad_norm
        """
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            if self.accelerator.sync_gradients:  # safe in gradient accumulation cases
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.config["trainer"]["max_grad_norm"]
                )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        """
        Calculates the gradient norm for logging.

        Args:
            norm_type (float | str | None): the order of the norm.
        Returns:
            total_norm (float): the calculated norm.
        """
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _progress(self, batch_idx):
        """
        Calculates the percentage of processed batch within the epoch.

        Args:
            batch_idx (int): the current batch index.
        Returns:
            progress (str): contains current step and percentage
                within the epoch.
        """
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx
        total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Abstract method. Should be defined in the nested Trainer Class.

        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        """
        Wrapper around the writer 'add_scalar' to log all metrics.

        Args:
            metric_tracker (MetricTracker): calculated metrics.
        """
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint_in_dir(self, dir_name):
        """
        Save the checkpoint in the requested directory name.
        Must be called only from the main process.

        If the multi-node setup does not share the filesystem,
        saving must be called from the *local* main process.

        Args:
            dir_name (str): name of the directory inside self.checkpoint_dir
        """
        full_checkpoint_dir = self.checkpoint_dir / dir_name
        full_checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Saving checkpoint: {full_checkpoint_dir} ...")

        arch = type(self.model).__name__
        trainer_state = {
            "arch": arch,
            "epoch": self._last_epoch,
            "global_step": self.global_step,
            "epoch_step": self.epoch_step,
            "seen_dataloader_batches": self.seen_dataloader_batches,
            "monitor_best": self.mnt_best,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        write_json(trainer_state, str(full_checkpoint_dir / "trainer_state.json"))

        # save accelerator state
        self.accelerator.save_state(output_dir=full_checkpoint_dir)

        # save model weights
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        # separately to avoid overwriting state weights
        model_weights_dir = full_checkpoint_dir / "model_weights"
        unwrapped_model.save_pretrained(
            model_weights_dir,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )

        # create symlink to the last checkpoint
        link_name = self.checkpoint_dir / "checkpoint-last"
        if link_name.exists() or link_name.is_symlink():
            if link_name.is_dir() and not link_name.is_symlink():
                self.logger.info("Deleting non-symlink last checkpoint ...")
                shutil.rmtree(link_name)
            else:
                link_name.unlink()
        os.symlink(dir_name, str(link_name), target_is_directory=True)

        if self.accelerator.is_main_process:
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(
                    full_checkpoint_dir, str(full_checkpoint_dir.parent)
                )

            if self.cfg_trainer.hf_push_to_hub:
                upload_folder(
                    repo_id=self.cfg_trainer.hf_repo_id,
                    folder_path=full_checkpoint_dir,
                    path_in_repo="checkpoint-last",  # rename
                    commit_message=f"Update latest checkpoint ({dir_name})",
                    delete_patterns="checkpoint-last/*",  # remove old data
                )

    def _save_checkpoint(self, save_best=False, only_best=False):
        """
        Save the checkpoints.

        If the multi-node setup does not share the filesystem,
        saving must be called from the *local* main process.
        Therefore, replace is_main_process here with is_local_main_process.

        Args:
            save_best (bool): if True, rename the saved checkpoint to
                'checkpoint-GlobalStepNumber'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'checkpoint-best'(do not duplicate the checkpoint as
                checkpoint-GlobalStepNumber)
        """
        if self.accelerator.is_main_process:
            dir_name = f"checkpoint-{self.global_step}"
            if save_best and only_best:
                dir_name = "checkpoint-best"
                self._save_checkpoint_in_dir(dir_name)
                return None
            self._save_checkpoint_in_dir(dir_name)
            if save_best:
                dir_name = "checkpoint-best"
                self._save_checkpoint_in_dir(dir_name)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').
        You can also resume from the remote HuggingFace repo.

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
                Use 'huggingface' to resume from the last checkpoint saved in
                'config.trainer.hf_repo_id'.
        """
        if resume_path == "huggingface":
            if self.accelerator.is_main_process:
                snapshot_download(
                    repo_id=self.cfg_trainer.hf_repo_id,
                    local_dir=self.checkpoint_dir,
                )
            resume_path = self.checkpoint_dir / "checkpoint-last"
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self.logger.info(f"Loading checkpoint: {resume_path} ...")

        trainer_state = read_json(str(resume_path / "trainer_state.json"))
        self.start_epoch = trainer_state["epoch"]
        self._last_epoch = trainer_state["epoch"]
        self.global_step = trainer_state["global_step"]
        self.epoch_step = trainer_state["epoch_step"]
        self.seen_dataloader_batches = trainer_state["seen_dataloader_batches"]

        if self.epoch_step == -1:
            # the saving was done at the end of epoch
            # we will start from the next one
            self.start_epoch += 1

        if self.accelerator.is_main_process:
            self.writer.set_step(self.global_step)

        self.mnt_best = trainer_state["monitor_best"]

        # load architecture params from checkpoint.
        if trainer_state["config"]["model"] != self.config["model"]:
            if self.accelerator.is_main_process:
                self.logger.warning(
                    "Warning: Architecture configuration given in the config file is different from that "
                    "of the checkpoint. This may yield an exception when state_dict is loaded."
                )
        self.accelerator.load_state(resume_path)

        if self.seen_dataloader_batches == 0:
            # to avoid repeating the dataloader state twice,
            # i.e., to avoid skipped_dataloader == train_dataloader (state-wise)
            skipped_dataloader = None
        else:
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Skipping first {self.seen_dataloader_batches} batches in the dataloader after resume."
                )
            skipped_dataloader = self.accelerator.skip_first_batches(
                self.train_dataloader, self.seen_dataloader_batches
            )

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
            )
        self.accelerator.wait_for_everyone()

        return skipped_dataloader
