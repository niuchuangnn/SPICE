# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR, RampedLR
from torch.optim.lr_scheduler import CosineAnnealingLR


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.solver.base_lr
        weight_decay = cfg.solver.weight_decay
        if "bias" in key:
            lr = cfg.solver.base_lr * cfg.solver.bias_lr_factor
            weight_decay = cfg.solver.weight_decay_bias

        if "prior_d" in key:
            lr = 0.0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if "type" in cfg.solver:
        opt_type = cfg.solver.type
    else:
        opt_type = "sgd"

    if opt_type == "sgd":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.solver.momentum)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif opt_type == "rmsprop":
        optimizer = torch.optim.RMSprop(params, lr=lr)
    else:
        raise TypeError

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    # return WarmupMultiStepLR(
    #     optimizer,
    #     cfg.solver.steps,
    #     cfg.solver.gamma,
    #     warmup_factor=cfg.solver.warmup_factor,
    #     warmup_iters=cfg.solver.warmup_iters,
    #     warmup_method=cfg.solver.warmup_method,
    # )
    # return CosineAnnealingLR(optimizer, T_max=cfg.solver.max_iter)
    return RampedLR(optimizer, cfg.solver.max_iter, cfg.solver.ramp_up_fraction, cfg.solver.ramp_down_fraction)
