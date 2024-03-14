from abc import abstractmethod

import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_

from src.logger.wandb import WanDBWriter
from src.metrics.tracker import MetricTracker
from src.utils.data_utils import inf_loop
from src.utils.io_utils import ROOT_PATH


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        logger,
        epoch_len=None,
        skip_oom=True,
        batch_transforms=None,
    ):
        self.is_train = True

        self.config = config
        cfg_trainer = config.trainer

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
        if epoch_len is None:
            # epoch-based training
            self.epoch_len = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len

        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }

        # define epochs
        self._last_epoch = 0  # required for saving on interruption
        self.start_epoch = 1
        self.epochs = cfg_trainer.n_epochs

        # configuration to monitor model performance and save best

        self.save_period = cfg_trainer.save_period  # checkpoint each save_period epochs
        self.monitor = cfg_trainer.get(
            "monitor", "off"
        )  # format: "mnt_mode mnt_metric"

        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        # setup visualization writer instance
        self.writer = WanDBWriter(config, self.logger)

        # define metrics
        self.metrics = metrics
        self.train_metrics = MetricTracker(
            *self.config.writer.loss_names,
            "grad_norm",
            *[m.name for m in self.metrics],
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *self.config.writer.loss_names,
            *[m.name for m in self.metrics],
            writer=self.writer,
        )

        # define checkpoint dir and init everything if required

        self.checkpoint_dir = (
            ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        )

        if config.trainer.get("resume_from") is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)

        if config.trainer.get("from_pretrained") is not None:
            self._from_pretrained(config.trainer.get("from_pretrained"))

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best, stop_process, not_improved_count = self._monitor_performance(
                log, not_improved_count
            )

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

            if stop_process:  # early_stop
                break

    def _monitor_performance(self, log, not_improved_count):
        best = False
        stop_process = False
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not,
                # according to specified metric(mnt_metric)
                if self.mnt_mode == "min":
                    improved = log[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == "max":
                    improved = log[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(
                    "Warning: Metric '{}' is not found. "
                    "Model performance monitoring is disabled.".format(self.mnt_metric)
                )
                self.mnt_mode = "off"
                improved = False

            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                self.logger.info(
                    "Validation performance didn't improve for {} epochs. "
                    "Training stops.".format(self.early_stop)
                )
                stop_process = True
        return best, stop_process, not_improved_count

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device
        """
        for tensor_for_device in self.config.trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
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
        if self.config["trainer"].get("max_grad_norm", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["max_grad_norm"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
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
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file

        :param pretrained_path: path to model state dict
        """
        pretrained_path = str(pretrained_path)
        self.logger.info("Loading model weights from: {} ...".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
