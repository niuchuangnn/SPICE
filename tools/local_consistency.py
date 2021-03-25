#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
    "--embedding",
    default="./results/stl10/embedding/feas_moco_512_l2.npy",
    type=str,
)


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_id)

    cfg.embedding = args.embedding

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

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

    state_dict = torch.load(cfg.model.pretrained)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict)
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
    feas_sim = torch.from_numpy(np.load(cfg.embedding))

    pred_labels = torch.cat(pred_labels).long().cpu().numpy()
    scores = torch.cat(scores_all).cpu()

    try:
        acc = calculate_acc(pred_labels, gt_labels)
    except:
        acc = -1

    nmi = calculate_nmi(pred_labels, gt_labels)
    ari = calculate_ari(pred_labels, gt_labels)

    print("ACC: {}, NMI: {}, ARI: {}".format(acc, nmi, ari))

    idx_select, labels_select = model(feas_sim=feas_sim, scores=scores, forward_type="local_consistency")

    gt_labels_select = gt_labels[idx_select]

    acc = calculate_acc(labels_select, gt_labels_select)
    print('ACC of local consistency: {}, number of samples: {}'.format(acc, len(gt_labels_select)))

    labels_correct = np.zeros([feas_sim.shape[0]]) - 100
    labels_correct[idx_select] = labels_select

    np.save("{}/labels_reliable_{:4f}_{}.npy".format(cfg.results.output_dir, acc, len(idx_select)), labels_correct)


if __name__ == '__main__':
    main()
