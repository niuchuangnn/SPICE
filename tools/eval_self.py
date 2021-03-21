import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from spice.config import Config
from spice.data.build_dataset import build_dataset
from spice.model.build_model_sim import build_model_sim
from spice.model.sim2sem import Sim2Sem
from spice.solver import make_lr_scheduler, make_optimizer
from spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from spice.utils.load_model_weights import load_model_weights
from matplotlib.pyplot import imsave
from PIL import Image
import matplotlib.pyplot as plt
import PIL


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/stl10/eval.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "--weight",
    default="./model_zoo/self_model_stl10.pth.tar",
    metavar="FILE",
    help="path to weight file",
    type=str,
)
parser.add_argument(
    "--all",
    default=1,
    type=int,
)
parser.add_argument(
    "--proto",
    default=0,
    type=int,
)
parser.add_argument(
    "--embedding",
    default="./results/stl10/embedding/feas_moco_512_l2.npy",
    type=str,
)


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    cfg.model.pretrained = args.weight
    cfg.proto = args.proto
    cfg.embedding = args.embedding
    cfg.all = args.all
    if cfg.all:
        cfg.data_test.split = "train+test"
        cfg.data_test.all = True
    else:
        cfg.data_test.split = "test"
        cfg.data_test.all = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_id)

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)
        if cfg.proto:
            mkdir("{}/proto".format(output_dir))

    output_config_path = os.path.join(output_dir, 'config.py')
    save_config(cfg, output_config_path)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if cfg.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if False: #cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg.copy()))
    else:
        # Simply call main_worker function
        args.gpu = 0
        cfg.gpu = 0
        main_worker(args.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    cfg.gpu = gpu

    # suppress printing if not master
    if cfg.multiprocessing_distributed and cfg.gpu != 0:
        def print_pass(*cfg):
            pass
        builtins.print = print_pass

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
    # create model
    model = Sim2Sem(**cfg.model)
    print(model)

    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optionally resume from a checkpoint
    # if cfg.model.pretrained is not None:
    #     load_model_weights(model, cfg.model.pretrained, cfg.model.model_type, cfg.model.head_id)

    state_dict = torch.load(cfg.model.pretrained)
    model.load_state_dict(state_dict)

    # state_dict = torch.load(cfg.model.pretrained)['state_dict']
    # state_dict_save = {}
    # for k in list(state_dict.keys()):
    #     if not k.startswith('module.head'):
    #         state_dict_save[k] = state_dict[k]
    #     # print(k)
    #     if k.startswith('module.head.head_{}'.format(cfg.model.head_id)):
    #         state_dict_save['module.head.head_0.{}'.format(k[len('module.head.head_{}.'.format(cfg.model.head_id))::])] = \
    #         state_dict[k]
    #
    # torch.save(state_dict_save, '{}/checkpoint_select.pth.tar'.format(cfg.results.output_dir))
    # model.load_state_dict(state_dict_save)

    # Load similarity model
    cudnn.benchmark = True

    # Data loading code
    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

    model.eval()

    num_heads = len(cfg.model.head.multi_heads)
    assert num_heads == 1
    gt_labels = []
    pred_labels = []
    scores_all = []
    # feas_sim = []

    for _, (images, _, labels, idx) in enumerate(val_loader):
        images = images.to(cfg.gpu, non_blocking=True)
        # print(images.shape)
        with torch.no_grad():
            scores = model(images, forward_type="sem")

        # feas_sim.append(embs)
        assert len(scores) == num_heads

        pred_idx = scores[0].argmax(dim=1)
        pred_labels.append(pred_idx)
        scores_all.append(scores[0])

        gt_labels.append(labels)

    gt_labels = torch.cat(gt_labels).long().cpu().numpy()
    # feas_sim = torch.cat(feas_sim, dim=0)

    pred_labels = torch.cat(pred_labels).long().cpu().numpy()
    scores = torch.cat(scores_all).cpu()

    try:
        acc = calculate_acc(pred_labels, gt_labels)
    except:
        acc = -1

    nmi = calculate_nmi(pred_labels, gt_labels)
    ari = calculate_ari(pred_labels, gt_labels)

    print("ACC: {}, NMI: {}, ARI: {}".format(acc, nmi, ari))

    if cfg.proto:
        data = val_loader.dataset.data
        feas_sim = np.load(cfg.embedding)
        feas_sim = torch.from_numpy(feas_sim)
        centers = model(feas_sim=feas_sim, scores=scores, forward_type="proto")

        sim_all = torch.einsum('nd,cd->nc', [feas_sim.cpu(), centers.cpu()])

        _, top_10 = torch.topk(sim_all, 10, 0)

        imgs = []
        for c in range(cfg.num_cluster):
            idx_c = top_10[:, c]
            img_c = data[idx_c, ...]
            imgs.append(img_c)
            for ii in range(10):
                img_c_ii = img_c[ii, ...].transpose([1, 2, 0])
                imsave('{}/proto/{}_{}.png'.format(cfg.results.output_dir, c, ii), img_c_ii)

            #     plt.figure()
            #     plt.imshow(img_c_ii)
            # plt.show()

        # imgs = np.concatenate(imgs, axis=0)

        for c in range(cfg.num_cluster):
            dataset_val.data = imgs[c]
            for i in range(len(dataset_val)):
                img, _, labels, idx = dataset_val[i]
                img = torch.unsqueeze(img, dim=0).to(cfg.gpu, non_blocking=True)
                with torch.no_grad():
                    fea_conv = model(img, forward_type="feature_only")
                fea_conv = fea_conv.reshape(512, 49)
                center = centers[c:c+1, :]
                sim_map = torch.einsum('nd,dm->nm', [center.cpu(), fea_conv.cpu()])
                sim_map = sim_map.reshape([7, 7])
                sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
                sim_map = sim_map.cpu().numpy()
                # print(sim_map.shape)

                img_c_ii = imgs[c][i, ...].transpose([1, 2, 0])

                sim_map = Image.fromarray(np.uint8(sim_map * 255))
                sim_map = sim_map.resize((img_c_ii.shape[1], img_c_ii.shape[0]), resample=PIL.Image.BILINEAR)
                sim_map = np.asarray(sim_map)

                att_mask = np.zeros_like(img_c_ii)
                att_mask[:, :, 0] = sim_map

                cmap = plt.get_cmap('jet')
                attMap = sim_map
                attMapV = cmap(attMap)
                attMapV = np.delete(attMapV, 3, 2) * 255

                attMap = 0.6 * img_c_ii + 0.4 * attMapV
                attMap = attMap.astype(np.uint8)
                imsave('{}/proto/{}_{}_att.png'.format(cfg.results.output_dir, c, i), attMap)

                # plt.figure()
                # plt.imshow(attMap)
                # plt.show()


if __name__ == '__main__':
    main()
