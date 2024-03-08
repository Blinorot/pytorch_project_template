import numpy as np
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
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)
