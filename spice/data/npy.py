from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
from torchvision.datasets.cifar import CIFAR10
import matplotlib.pyplot as plt


class NPYEMB(CIFAR10):
    """
    """

    def __init__(self, root, show=False, transform1=None, transform2=None,
                 embedding=None):
        self.root = os.path.expanduser(root)
        self.transform1 = transform1
        self.transform2 = transform2

        self.show = show
        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None

        # now load the picked numpy arrays
        self.data = np.load("{}/data.npy".format(self.root))
        self.labels = np.load("{}/label.npy".format(self.root))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.embedding is not None:
            emb = self.embedding[index]
        else:
            emb = None

        if self.transform1 is not None:
            img_trans1 = self.transform1(img)
        else:
            img_trans1 = img

        if self.transform2 is not None:
            img_trans2 = self.transform2(img)
        else:
            img_trans2 = img

        if self.show:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            img_trans1 = img_trans1.numpy().transpose([1, 2, 0]) * std + mean
            # img_trans1 = img_trans1.numpy().transpose([1, 2, 0])
            # img_trans1 = (img_trans1 - img_trans1.min()) / (img_trans1.max() - img_trans1.min())
            plt.figure()
            plt.imshow(img_trans1)

            img_trans2 = img_trans2.numpy().transpose([1, 2, 0]) * std + mean
            plt.figure()
            plt.imshow(img_trans2)
            plt.show()

        if emb is not None:
            return img_trans1, img_trans2, emb, target, index
        else:
            return img_trans1, img_trans2, target, index

    def __len__(self):
        return self.data.shape[0]


class NPY(CIFAR10):
    """
    """

    def __init__(self, root, show=False, transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform

        self.show = show

        # now load the picked numpy arrays
        self.data = np.load("{}/data.npy".format(self.root))
        self.labels = np.load("{}/label.npy".format(self.root))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.show:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            if isinstance(img, list):
                img_show = img[0]
            else:
                img_show = img
            img_show = img_show.numpy().transpose([1, 2, 0]) * std + mean
            plt.figure()
            plt.imshow(img_show)
            plt.show()

        return img, target

    def __len__(self):
        return self.data.shape[0]
