# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch
import math
import numpy as np


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class RampedLR(torch.optim.lr_scheduler._LRScheduler):
    r"""

    """

    def __init__(self, optimizer, iteration_count, ramp_up_fraction, ramp_down_fraction, last_epoch=-1):
        self.iteration_count = iteration_count
        self.ramp_up_fraction = ramp_up_fraction
        self.ramp_down_fraction = ramp_down_fraction

        if ramp_up_fraction > 0.0:
            self.ramp_up_end_iter = iteration_count * ramp_up_fraction
        else:
            self.ramp_up_end_iter = None

        if ramp_down_fraction > 0.0:
            self.ramp_down_start_iter = iteration_count * (1 - ramp_down_fraction)
        else:
            self.ramp_down_start_iter = None

        super(RampedLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.ramp_up_end_iter is not None:
            if self.last_epoch <= self.ramp_up_end_iter:
                return [base_lr * (0.5 - np.cos(((self.last_epoch / self.ramp_up_fraction) / self.iteration_count) * np.pi)/2)
                        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)]

        if self.ramp_down_fraction is not None:
            if self.last_epoch >= self.ramp_down_start_iter:
                return [base_lr * (0.5 + np.cos((((self.last_epoch - self.ramp_down_start_iter) / self.ramp_down_fraction) / self.iteration_count) * np.pi)/2)**2
                        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs)]

        return [group['lr'] for group in self.optimizer.param_groups]
