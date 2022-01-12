from spice.model.feature_modules.resnet_all import resnet34


class build_ResNet34:
    def __init__(self, **kwargs):
        pass

    def build(self, num_classes):
        return resnet34(num_classes=num_classes)