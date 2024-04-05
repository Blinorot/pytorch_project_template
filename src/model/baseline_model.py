from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    def __init__(self, n_feats, n_class, fc_hidden=512):
        super().__init__()
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, data_object, **batch):
        return {"logits": self.net(data_object)}

    def __str__(self):
        """
        Model prints with number of parameters
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
