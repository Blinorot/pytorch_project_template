from huggingface_hub.dataclasses import strict
from torch import nn
from torch.nn import Sequential
from transformers import PreTrainedConfig, PreTrainedModel


@strict(accept_kwargs=True)
class BaselineConfig(PreTrainedConfig):
    model_type = "baseline"

    # it is important to provide default values for all arguments
    def __init__(self, n_feats=1024, n_class=2, fc_hidden=512, **kwargs):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
            fc_hidden (int): number of hidden features.
        """
        super().__init__(**kwargs)
        self.n_feats = n_feats
        self.n_class = n_class
        self.fc_hidden = fc_hidden


class BaselineModel(PreTrainedModel):
    """
    Simple MLP
    """

    config_class = BaselineConfig

    def __init__(self, config, **kwargs):
        """
        Args:
            config (BaselineConfig): configuration.
        """
        super().__init__(config, **kwargs)

        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(
                in_features=self.config.n_feats, out_features=self.config.fc_hidden
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.config.fc_hidden, out_features=self.config.fc_hidden
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.config.fc_hidden, out_features=self.config.n_class
            ),
        )

        # must be called to fully initialize PreTrainedModel
        # required for enabling .from_pretrained()
        # see https://huggingface.co/docs/transformers/main/modeling_rules
        self.post_init()

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input image.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(img.flatten(1))}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
