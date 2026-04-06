from transformers import AutoConfig, AutoModel

from src.model.baseline_model import BaselineConfig, BaselineModel


def register_models():
    # Register all models and their configs
    AutoConfig.register(BaselineConfig.model_type, BaselineConfig)
    AutoModel.register(BaselineConfig, BaselineModel)
