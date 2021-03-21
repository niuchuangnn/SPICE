from spice.data.stl10 import STL10
from spice.data.transformations import get_train_transformations
from spice.data.stl10_embedding import STL10EMB
from spice.data.cifar import CIFAR10, CIFAR20
from spice.data.imagenet import ImageNetSubEmb, ImageNetSubEmbLMDB, TImageNetEmbLMDB
from spice.data.npy import NPYEMB


def build_dataset(data_cfg):
    type = data_cfg.type

    dataset = None

    train_trans1 = get_train_transformations(data_cfg.trans1)
    train_trans2 = get_train_transformations(data_cfg.trans2)

    if type == "stl10":
        dataset = STL10(root=data_cfg.root_folder,
                        split=data_cfg.split,
                        show=data_cfg.show,
                        transform1=train_trans1,
                        transform2=train_trans2,
                        download=False)
    elif type == "stl10_emb":
        dataset = STL10EMB(root=data_cfg.root_folder,
                           split=data_cfg.split,
                           show=data_cfg.show,
                           transform1=train_trans1,
                           transform2=train_trans2,
                           download=False,
                           embedding=data_cfg.embedding)
    elif type == "npy_emb":
        dataset = NPYEMB(root=data_cfg.root,
                         show=data_cfg.show,
                         transform1=train_trans1,
                         transform2=train_trans2,
                         embedding=data_cfg.embedding)
    elif type == "cifar10":
        dataset = CIFAR10(root=data_cfg.root_folder,
                          all=data_cfg.all,
                          train=data_cfg.train,
                          transform1=train_trans1,
                          transform2=train_trans2,
                          target_transform=None,
                          download=False,
                          embedding=data_cfg.embedding,
                          show=data_cfg.show,
                          )
    elif type == "cifar100":
        dataset = CIFAR20(root=data_cfg.root_folder,
                          all=data_cfg.all,
                          train=data_cfg.train,
                          transform1=train_trans1,
                          transform2=train_trans2,
                          target_transform=None,
                          download=False,
                          embedding=data_cfg.embedding,
                          show=data_cfg.show,
                          )
    elif type == 'imagenet':
        dataset = ImageNetSubEmb(subset_file=data_cfg.subset_file,
                                 embedding=data_cfg.embedding,
                                 split=data_cfg.split,
                                 transform1=train_trans1,
                                 transform2=train_trans2)
    elif type == 'imagenet_lmdb':
        dataset = ImageNetSubEmbLMDB(lmdb_file=data_cfg.lmdb_file,
                                     meta_info_file=data_cfg.meta_info_file,
                                     embedding=data_cfg.embedding,
                                     split=data_cfg.split,
                                     transform1=train_trans1,
                                     transform2=train_trans2,
                                     resize=data_cfg.resize)
    elif type == 'timagenet_lmdb':
        dataset = TImageNetEmbLMDB(lmdb_file=data_cfg.lmdb_file,
                                   meta_info_file=data_cfg.meta_info_file,
                                   embedding=data_cfg.embedding,
                                   transform1=train_trans1,
                                   transform2=train_trans2)
    else:
        assert TypeError

    return dataset