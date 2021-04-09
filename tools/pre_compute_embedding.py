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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari
from spice.utils.load_model_weights import load_model_weights
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/cifar10/embedding.py",
    metavar="FILE",
    help="path to config file",
    type=str,
)


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
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
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg.copy()))
    else:
        # Simply call main_worker function
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
    model_sim = build_model_sim(cfg.model_sim)
    print(model_sim)

    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model_sim.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int((cfg.workers + ngpus_per_node - 1) / ngpus_per_node)
            model_sim = torch.nn.parallel.DistributedDataParallel(model_sim, device_ids=[cfg.gpu])
        else:
            model_sim.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_sim = torch.nn.parallel.DistributedDataParallel(model_sim)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model_sim = model_sim.cuda(cfg.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # Load similarity model

    if cfg.model_sim.pretrained is not None:
        load_model_weights(model_sim, cfg.model_sim.pretrained, cfg.model_sim.model_type)

    cudnn.benchmark = True

    dataset_val = build_dataset(cfg.data_test)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=cfg.batch_size, shuffle=False, num_workers=1)

    model_sim.eval()

    pool = nn.AdaptiveAvgPool2d(1)

    feas_sim = []
    for _, (images, _, labels, idx) in enumerate(val_loader):
        images = images.to(cfg.gpu, non_blocking=True)
        print(images.shape)
        with torch.no_grad():
            feas_sim_i = model_sim(images)
            if len(feas_sim_i.shape) == 4:
                feas_sim_i = pool(feas_sim_i)
                feas_sim_i = torch.flatten(feas_sim_i, start_dim=1)
            feas_sim_i = nn.functional.normalize(feas_sim_i, dim=1)
            feas_sim.append(feas_sim_i.cpu())

    feas_sim = torch.cat(feas_sim, dim=0)
    feas_sim = feas_sim.numpy()

    np.save("{}/feas_moco_512_l2.npy".format(cfg.results.output_dir), feas_sim)


if __name__ == '__main__':
    main()
