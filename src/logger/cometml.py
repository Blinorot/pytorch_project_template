from datetime import datetime

import numpy as np
import pandas as pd


class CometMLWriter:
    """
    Class for experiment tracking via CometML.

    See https://www.comet.com/docs/v2/.
    """

    def __init__(
        self,
        logger,
        project_config,
        project_name,
        workspace=None,
        run_id=None,
        run_name=None,
        mode="online",
        **kwargs,
    ):
        """
        API key is expected to be provided by the user in the terminal.

        Args:
            logger (Logger): logger that logs output.
            project_config (dict): config for the current experiment.
            project_name (str): name of the project inside experiment tracker.
            workspace (str | None): name of the workspace inside experiment
                tracker. Used if you work in a team.
            run_id (str | None): the id of the current run.
            run_name (str | None): the name of the run. If None, random name
                is given.
            mode (str): if online, log data to the remote server. If
                offline, log locally.
        """
        try:
            import comet_ml

            comet_ml.login()

            self.run_id = run_id

            resume = False
            if project_config["trainer"].get("resume_from") is not None:
                resume = True

            if resume:
                if mode == "offline":
                    exp_class = comet_ml.ExistingOfflineExperiment
                else:
                    exp_class = comet_ml.ExistingExperiment

                self.exp = exp_class(experiment_key=self.run_id)
            else:
                if mode == "offline":
                    exp_class = comet_ml.OfflineExperiment
                else:
                    exp_class = comet_ml.Experiment

                self.exp = exp_class(
                    project_name=project_name,
                    workspace=workspace,
                    experiment_key=self.run_id,
                    log_code=kwargs.get("log_code", False),
                    log_graph=kwargs.get("log_graph", False),
                    auto_metric_logging=kwargs.get("auto_metric_logging", False),
                    auto_param_logging=kwargs.get("auto_param_logging", False),
                )
                self.exp.set_name(run_name)
                self.exp.log_parameters(parameters=project_config)

            self.comel_ml = comet_ml

        except ImportError:
            logger.warning("For use comet_ml install it via \n\t pip install comet_ml")

        self.step = 0
        # the mode is usually equal to the current partition name
        # used to separate Partition1 and Partition2 metrics
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        """
        Define current step and mode for the tracker.

        Calculates the difference between method calls to monitor
        training/evaluation speed.

        Args:
            step (int): current step.
            mode (str): current mode (partition name).
        """
        self.mode = mode
        previous_step = self.step
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar(
                "steps_per_sec", (self.step - previous_step) / duration.total_seconds()
            )
            self.timer = datetime.now()

    def _object_name(self, object_name):
        """
        Update object_name (scalar, image, etc.) with the
        current mode (partition name). Used to separate metrics
        from different partitions.

        Args:
            object_name (str): current object name.
        Returns:
            object_name (str): updated object name.
        """
        return f"{object_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        """
        Log checkpoints to the experiment tracker.

        The checkpoints will be available in the Assets & Artifacts section
        inside the models/checkpoints directory.

        Args:
            checkpoint_path (str): path to the checkpoint file.
            save_dir (str): path to the dir, where checkpoint is saved.
        """
        # For comet, save dir is not required
        # It is kept for consistency with WandB
        self.exp.log_model(
            name="checkpoints", file_or_folder=checkpoint_path, overwrite=True
        )

    def add_scalar(self, scalar_name, scalar):
        """
        Log a scalar to the experiment tracker.

        Args:
            scalar_name (str): name of the scalar to use in the tracker.
            scalar (float): value of the scalar.
        """
        self.exp.log_metrics(
            {
                self._object_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, scalars):
        """
        Log several scalars to the experiment tracker.

        Args:
            scalars (dict): dict, containing scalar name and value.
        """
        self.exp.log_metrics(
            {
                self._object_name(scalar_name): scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_image(self, image_name, image):
        """
        Log an image to the experiment tracker.

        Args:
            image_name (str): name of the image to use in the tracker.
            image (Path | Tensor | ndarray | list[tuple] | Image): image
                in the CometML-friendly format.
        """
        self.exp.log_image(
            image_data=image, name=self._object_name(image_name), step=self.step
        )

    def add_audio(self, audio_name, audio, sample_rate=None):
        """
        Log an audio to the experiment tracker.

        Args:
            audio_name (str): name of the audio to use in the tracker.
            audio (Path | ndarray): audio in the CometML-friendly format.
            sample_rate (int): audio sample rate.
        """
        audio = audio.detach().cpu().numpy().T
        self.exp.log_audio(
            file_name=self._object_name(audio_name),
            audio_data=audio,
            sample_rate=sample_rate,
            step=self.step,
        )

    def add_text(self, text_name, text):
        """
        Log text to the experiment tracker.

        Args:
            text_name (str): name of the text to use in the tracker.
            text (str): text content.
        """
        self.exp.log_text(
            text=text, step=self.step, metadata={"name": self._object_name(text_name)}
        )

    def add_histogram(self, hist_name, values_for_hist, bins=None):
        """
        Log histogram to the experiment tracker.

        Args:
            hist_name (str): name of the histogram to use in the tracker.
            values_for_hist (Tensor): array of values to calculate
                histogram of.
            bins (int | str): the definition of bins for the histogram.
        """
        # For comet, bins argument is not required
        # It is kept for consistency with WandB

        values_for_hist = values_for_hist.detach().cpu().numpy()

        # np_hist = np.histogram(values_for_hist, bins=bins)
        # if np_hist[0].shape[0] > 512:
        #     np_hist = np.histogram(values_for_hist, bins=512)

        self.exp.log_histogram_3d(
            values=values_for_hist, name=self._object_name(hist_name), step=self.step
        )

    def add_table(self, table_name, table: pd.DataFrame):
        """
        Log table to the experiment tracker.

        Args:
            table_name (str): name of the table to use in the tracker.
            table (DataFrame): table content.
        """
        self.exp.set_step(self.step)
        # log_table does not support step directly
        self.exp.log_table(
            filename=self._object_name(table_name) + ".csv",
            tabular_data=table,
            headers=True,
        )

    def add_images(self, image_names, images):
        raise NotImplementedError()

    def add_pr_curve(self, curve_name, curve):
        raise NotImplementedError()

    def add_embedding(self, embedding_name, embedding):
        raise NotImplementedError()
