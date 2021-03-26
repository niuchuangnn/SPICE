from .convnet import ConvNet
from .mlp import MLP
from .cluster_resnet import ClusterResNet
from .resnet_stl import resnet18
from .resnet_cifar import resnet18_cifar
from .imagenet import ImageNet


def build_feature_module(fea_cfg_ori):
    fea_cfg = fea_cfg_ori.copy()
    fea_type = fea_cfg.pop("type")
    if fea_type == "mlp":
        return MLP(**fea_cfg)
    elif fea_type == "convnet":
        return ConvNet(**fea_cfg)
    elif fea_type == "clusterresnet":
        return ClusterResNet(**fea_cfg)
    elif fea_type == "resnet18":
        return resnet18(**fea_cfg)
    elif fea_type == "resnet18_cifar":
        return resnet18_cifar(**fea_cfg)
    elif fea_type == "imagenet":
        return ImageNet(**fea_cfg)
    else:
        raise TypeError