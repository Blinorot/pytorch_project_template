from datetime import datetime

import numpy as np
import pandas as pd


class CometMLWriter:
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
                    log_code=False,
                    log_graph=False,
                    auto_metric_logging=False,
                    auto_param_logging=False,
                )
                self.exp.set_name(run_name)
                self.exp.log_parameters(parameters=project_config)

            self.comel_ml = comet_ml

        except ImportError:
            logger.warning("For use comet_ml install it via \n\t pip install comet_ml")

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
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

    def _scalar_name(self, scalar_name):
        return f"{scalar_name}_{self.mode}"

    def add_checkpoint(self, checkpoint_path, save_dir):
        # For comet, save dir is not required
        # It is kept for consistency with WandB
        self.exp.log_model(
            name="checkpoints", file_or_folder=checkpoint_path, overwrite=True
        )

    def add_scalar(self, scalar_name, scalar):
        self.exp.log_metrics(
            {
                self._scalar_name(scalar_name): scalar,
            },
            step=self.step,
        )

    def add_scalars(self, tag, scalars):
        self.exp.log_metrics(
            {
                f"{scalar_name}_{tag}_{self.mode}": scalar
                for scalar_name, scalar in scalars.items()
            },
            step=self.step,
        )

    def add_image(self, scalar_name, image):
        self.exp.log_image(
            image_data=image, name=self._scalar_name(scalar_name), step=self.step
        )

    def add_audio(self, scalar_name, audio, sample_rate=None):
        audio = audio.detach().cpu().numpy().T
        self.exp.log_audio(
            file_name=self._scalar_name(scalar_name),
            audio_data=audio,
            sample_rate=sample_rate,
            step=self.step,
        )

    def add_text(self, scalar_name, text):
        self.exp.log_text(
            text=text, step=self.step, metadata={"name": self._scalar_name(scalar_name)}
        )

    def add_histogram(self, scalar_name, hist, bins=None):
        hist = hist.detach().cpu().numpy()
        np_hist = np.histogram(hist, bins=bins)
        if np_hist[0].shape[0] > 512:
            np_hist = np.histogram(hist, bins=512)

        self.exp.log_histogram_3d(
            values=hist, name=self._scalar_name(scalar_name), step=self.step
        )

    def add_table(self, table_name, table: pd.DataFrame):
        self.exp.set_step(self.step)
        # log_table does not support step directly
        self.exp.log_table(
            filename=self._scalar_name(table_name) + ".csv",
            tabular_data=table,
            headers=True,
        )

    def add_images(self, scalar_name, images):
        raise NotImplementedError()

    def add_pr_curve(self, scalar_name, scalar):
        raise NotImplementedError()

    def add_embedding(self, scalar_name, scalar):
        raise NotImplementedError()
