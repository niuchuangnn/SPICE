# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F


class RFMModel(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, net_builder, sup_loss_type='sat', labels=None, num_classes=10, momentum=0.9, es=40):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(RFMModel, self).__init__()

        self.model = net_builder(num_classes=num_classes)
        self.sup_loss_type = sup_loss_type
        if self.sup_loss_type == 'sat':
            soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float)
            soft_labels[torch.arange(labels.shape[0]), labels] = 1
            self.register_buffer("soft_labels", soft_labels)
            self.momentum = momentum
            self.es = es


    @torch.no_grad()
    def _momentum_update_soft_labels(self, logits, index):
        """
        Momentum update of the soft labels
        """
        prob = F.softmax(logits.detach(), dim=1)
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

    def forward(self, inputs, num_lb=None, it=None, sup_index=None):
        """

        """
        logits = self.model(inputs)

        if self.training:
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
            del logits

            sup_soft_labels = None

            if self.sup_loss_type == 'sat' and it > self.es:
                # obtain prob, then update running avg

                self._momentum_update_soft_labels(logits_x_lb, sup_index)
                sup_soft_labels = self.soft_labels[sup_index]

            return logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, sup_soft_labels

        return logits


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def build_rfm(net_builder, sup_loss_type='sat', labels=None, num_classes=10, momentum=0.9, es=40):
    return RFMModel(net_builder, sup_loss_type=sup_loss_type, labels=labels, num_classes=num_classes, momentum=momentum, es=es)


class RMFBuilder:
    def __init__(self, net_builder, sup_loss_type='sat', labels=None, num_classes=10, momentum=0.9, es=40):
        self.net_builder = net_builder
        self.sup_loss_type = sup_loss_type
        self.labels = labels
        self.num_classes = num_classes
        self.momentum = momentum
        self.es = es

    def build_model(self):
        return RFMModel(self.net_builder, sup_loss_type=self.sup_loss_type,
                        labels=self.labels, num_classes=self.num_classes,
                        momentum=self.momentum, es=self.es)