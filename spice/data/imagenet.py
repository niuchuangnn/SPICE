"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from torchvision import transforms as tf
from glob import glob
import numpy as np
import pickle
import lmdb


def _get_keys_shapes_targets_pickle(meta_info_file):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(meta_info_file, 'rb'))
    keys = meta_info['keys']
    shapes = meta_info['shapes']
    targets = meta_info['targets']
    return keys, shapes, targets


def _read_img_lmdb(env, key, shape, dtype=np.uint8):
    with env.begin(write=False) as txn:
        buf = txn.get("{}".format(key).encode('ascii'))
    img_noise_flat = np.frombuffer(buf, dtype=dtype)
    H, W, C = shape
    img_noise = img_noise_flat.reshape(H, W, C)
    return img_noise


class ImageNet(datasets.ImageFolder):
    def __init__(self, root, split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, 'ILSVRC2012_img_%s' % (split)),
                                       transform=None)
        self.transform = transform
        self.split = split
        self.resize = tf.Resize(256)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img


class ImageNetSubset(data.Dataset):
    def __init__(self, subset_file, root, split='train',
                 transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, 'ILSVRC2012_img_%s' % (split))
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i))
        self.imgs = imgs
        self.classes = class_names

        # Resize
        self.resize = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}

        return out


class ImageNetSubEmb(data.Dataset):
    def __init__(self, subset_file, embedding, root, split='train',
                 transform1=None, transform2=None,):
        super(ImageNetSubEmb, self).__init__()

        self.root = os.path.join(root, 'ILSVRC2012_img_%s' % (split))
        self.transform1 = transform1
        self.transform2 = transform2
        self.split = split
        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i))
        self.imgs = imgs
        self.classes = class_names

        # Resize
        self.resize = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        # im_size = img.size
        img = self.resize(img)
        # class_name = self.classes[target]

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

        if emb is not None:
            return img_trans1, img_trans2, emb, target, index
        else:
            return img_trans1, img_trans2, target, index


class ImageNetSubEmbLMDB(data.Dataset):
    def __init__(self, lmdb_file, meta_info_file, embedding, split='train',
                 transform1=None, transform2=None, resize=256):
        super(ImageNetSubEmbLMDB, self).__init__()

        self.keys, self.shapes, self.targets = _get_keys_shapes_targets_pickle(meta_info_file)

        # self.data_env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)
        self.lmdb_file = lmdb_file
        self.data_env = None

        self.transform1 = transform1
        self.transform2 = transform2
        self.split = split
        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None

        # Resize
        self.resize = tf.Resize(resize)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img)
        return img

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        if self.data_env is None:
            self.data_env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

        img_shape = [int(s) for s in self.shapes[index].split('_')]
        img = _read_img_lmdb(self.data_env, self.keys[index], img_shape)
        img = Image.fromarray(img)
        target = int(self.targets[index])

        # path, target = self.imgs[index]
        # with open(path, 'rb') as f:
        #     img = Image.open(f).convert('RGB')

        # im_size = img.size
        img = self.resize(img)
        # class_name = self.classes[target]

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

        if False:
            import matplotlib.pyplot as plt
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            img_trans1_show = img_trans1.numpy().transpose([1, 2, 0]) * std + mean
            # img_trans1 = img_trans1.numpy().transpose([1, 2, 0])
            # img_trans1 = (img_trans1 - img_trans1.min()) / (img_trans1.max() - img_trans1.min())
            plt.figure()
            plt.imshow(img_trans1_show)

            img_trans2_show = img_trans2.numpy().transpose([1, 2, 0]) * std + mean
            plt.figure()
            plt.imshow(img_trans2_show)
            plt.show()

        if emb is not None:
            return img_trans1, img_trans2, emb, target, index
        else:
            return img_trans1, img_trans2, target, index

class TImageNetEmbLMDB(data.Dataset):
    def __init__(self, lmdb_file, meta_info_file, embedding,
                 transform1=None, transform2=None,):
        super(TImageNetEmbLMDB, self).__init__()

        self.keys, self.shapes, self.targets = _get_keys_shapes_targets_pickle(meta_info_file)

        # self.data_env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)
        self.lmdb_file = lmdb_file
        self.data_env = None

        self.transform1 = transform1
        self.transform2 = transform2
        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        if self.data_env is None:
            self.data_env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

        img_shape = [int(s) for s in self.shapes[index].split('_')]
        img = _read_img_lmdb(self.data_env, self.keys[index], img_shape)
        img = Image.fromarray(img)
        target = int(self.targets[index])

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

        if False:
            import matplotlib.pyplot as plt
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            img_trans1_show = img_trans1.numpy().transpose([1, 2, 0]) * std + mean
            # img_trans1 = img_trans1.numpy().transpose([1, 2, 0])
            # img_trans1 = (img_trans1 - img_trans1.min()) / (img_trans1.max() - img_trans1.min())
            plt.figure()
            plt.imshow(img_trans1_show)

            img_trans2_show = img_trans2.numpy().transpose([1, 2, 0]) * std + mean
            plt.figure()
            plt.imshow(img_trans2_show)
            plt.show()

        if emb is not None:
            return img_trans1, img_trans2, emb, target, index
        else:
            return img_trans1, img_trans2, target, index


class ImageNetLMDB(data.Dataset):
    def __init__(self, lmdb_file, meta_info_file, transform=None, resize=None):
        super(ImageNetLMDB, self).__init__()

        self.keys, self.shapes, self.targets = _get_keys_shapes_targets_pickle(meta_info_file)

        # self.data_env = lmdb.open(lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)
        self.lmdb_file = lmdb_file
        self.data_env = None

        self.transform = transform

        # Resize
        if resize is not None:
            self.resize = tf.Resize(resize)
        else:
            self.resize = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        if self.data_env is None:
            self.data_env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)

        img_shape = [int(s) for s in self.shapes[index].split('_')]
        img = _read_img_lmdb(self.data_env, self.keys[index], img_shape)
        img = Image.fromarray(img)
        target = int(self.targets[index])

        if self.resize is not None:
            img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        if False:
            import matplotlib.pyplot as plt
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            img_trans1_show = img_trans1.numpy().transpose([1, 2, 0]) * std + mean
            plt.figure()
            plt.imshow(img_trans1_show)

            plt.show()

        return img, target


if __name__ == "__main__":
    # dataset = ImageNetSubEmb('./datasets/imagenet_subsets/imagenet_50.txt', embedding=None)
    # dataset = ImageNetSubEmbLMDB(lmdb_file='/media/niuchuang/Storage/DataSets/ImageNet/imagenet50',
    #                              meta_info_file='/media/niuchuang/Storage/DataSets/ImageNet/imagenet50_meta_info.pkl',
    #                              embedding='/media/niuchuang/Storage/ModelResults/sim2sem/imagenet50/fea_moco/feas_imagenet50_l2.npy')
    # num_data = len(dataset)
    # import matplotlib.pyplot as plt
    #
    # for i in range(num_data):
    #     img1, img2, emb, target, idx = dataset[i]
    #     plt.figure()
    #     plt.imshow(img1)
    #     plt.show()
    #     pass
    dataset = ImageNetLMDB(lmdb_file='/media/niuchuang/Storage/DataSets/ImageNet/imagenet10_lmdb',
                           meta_info_file='/media/niuchuang/Storage/DataSets/ImageNet/imagenet10_lmdb_meta_info.pkl',
                           resize=96)
    num_data = len(dataset)
    import matplotlib.pyplot as plt

    for i in range(num_data):
        img, target = dataset[i]
        plt.figure()
        plt.imshow(img)
        plt.show()