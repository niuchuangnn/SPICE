from .feature_modules import build_feature_module


def build_model_sim(cfg):
    return build_feature_module(cfg)