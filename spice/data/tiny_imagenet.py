from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.datasets.cifar import CIFAR10
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder


class TinyImageNet(ImageFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            is_valid_file=None,
    ):
        super(TinyImageNet, self).__init__(root,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


if __name__ == "__main__":
    dataset = TinyImageNet(root='/media/niuchuang/Storage/DataSets/tiny-imagenet50-200/train')
    num_dataset = len(dataset)

    for i in range(num_dataset):
        data, label = dataset[i]
        plt.figure()
        pass