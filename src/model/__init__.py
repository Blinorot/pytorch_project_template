from transformers import AutoConfig, AutoModel

from src.model.baseline_model import BaselineConfig, BaselineModel
from src.model.resnet import MyResNetConfig, MyResNetModel


def register_models():
    # Register all models and their configs
    AutoConfig.register(BaselineConfig.model_type, BaselineConfig)
    AutoModel.register(BaselineConfig, BaselineModel)

    AutoConfig.register(MyResNetConfig.model_type, MyResNetConfig)
    AutoModel.register(MyResNetConfig, MyResNetModel)
