import torch.nn as nn
from .heads import build_head
from .feature_modules import build_feature_module
import torch


class Sim2Sem(nn.Module):

    def __init__(self, feature, head, freeze_conv=False, **kwargs):
        super(Sim2Sem, self).__init__()
        self.feature_module = build_feature_module(feature)
        self.head = build_head(head)
        if freeze_conv:
            for name, param in self.feature_module.named_parameters():
                param.requires_grad = False

    def forward(self, images=None, target=None, forward_type="sem", feas_sim=None, scores=None, epoch=None, return_centers=False):

        if forward_type not in ["sim2sem", "proto", "local_consistency"]:
            if isinstance(images, list):
                fea = []
                # for image in images:
                #     fea.append(self.feature_module(image))
                num_heads = len(images)
                num_each = images[0].shape[0]
                image = torch.cat(images, dim=0)
                fea_all = self.feature_module(image)
                for h in range(num_heads):
                    s = h*num_each
                    e = s + num_each
                    fea.append(fea_all[s:e, ...])
            else:
                fea = self.feature_module(images)

        if forward_type == "sem":
            return self.head.forward(fea)
        elif forward_type == "sim2sem":
            return self.head.select_samples(feas_sim, scores, epoch)
        elif forward_type == "local_consistency":
            return self.head.local_consistency(feas_sim, scores)
        elif forward_type == "proto":
            return self.head.compute_cluster_proto(feas_sim, scores)
        elif forward_type == "loss":
            return self.head.loss(fea, target)
        elif forward_type == "feature_only":
            return fea
        else:
            raise TypeError
