from .resnet_all import resnet50, resnet34
import torch.nn as nn


def ImageNet(num_classes, feature_only=False, **kwargs):
    model = resnet50(num_classes=num_classes, feature_only=feature_only)
    if not feature_only:
        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    return model


def ResNet34(num_classes, feature_only=False, **kwargs):
    model = resnet34(num_classes=num_classes, feature_only=feature_only)
    return model