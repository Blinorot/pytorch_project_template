import torch
import torchvision
from huggingface_hub.dataclasses import strict
from torch import nn
from transformers import PreTrainedConfig, PreTrainedModel


@strict(accept_kwargs=True)
class MyResNetConfig(PreTrainedConfig):
    model_type = "myresnet"  # cannot just use resnet since it is used by HF

    # it is important to provide default values for all arguments
    def __init__(self, input_channels=1, n_class=10, use_pretrained=True, **kwargs):
        """
        Args:
            input_channels (int): number of input_channels.
            n_class (int): number of classes.
            use_pretrained (bool): whether to use pretrained resnet weights.
        """
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.n_class = n_class
        self.use_pretrained = use_pretrained


class MyResNetModel(PreTrainedModel):
    """
    Wrapper over ResNet18 from torchvision
    """

    config_class = MyResNetConfig

    def __init__(self, config):
        """
        Args:
            config (MyResNetConfig): configuration.
        """
        super().__init__(config=config)

        if self.config.use_pretrained:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        self.resnet = torchvision.models.resnet18(weights=weights)
        self.resnet.conv1 = nn.Conv2d(
            self.config.input_channels,
            self.resnet.conv1.out_channels,
            self.resnet.conv1.kernel_size,
            self.resnet.conv1.stride,
            self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias is not None,
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.config.n_class)

        # must be called to fully initialize PreTrainedModel
        # required for enabling .from_pretrained()
        # see https://huggingface.co/docs/transformers/main/modeling_rules
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        # use torch init
        # prevents .post_init from overriding pretrained weights.
        pass

    def forward(self, img, **batch):
        """
        Model forward method.

        Args:
            img (Tensor): input img.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.resnet(img)}

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
