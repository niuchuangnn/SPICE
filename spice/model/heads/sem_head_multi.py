import torch
import torch.nn as nn
from ..feature_modules.build_feature_module import build_feature_module

import numpy as np
import torch.nn.functional as F
from spice.model.heads import build_head
from spice.model.heads.sem_head import SemHead

import matplotlib.pyplot as plt


class SemHeadMulti(nn.Module):
    def __init__(self, multi_heads, ratio_confident=0.90, num_neighbor=100, score_th=0.99, **kwargs):

        super(SemHeadMulti, self).__init__()

        self.num_heads = len(multi_heads)
        self.num_cluster = multi_heads[0].num_cluster
        self.center_ratio = multi_heads[0].center_ratio
        self.num_neighbor = num_neighbor
        self.ratio_confident = ratio_confident
        self.score_th = score_th
        for h in range(self.num_heads):
            head_h = SemHead(**multi_heads[h])
            self.__setattr__("head_{}".format(h), head_h)

    def local_consistency(self, feas_sim, scores):
        labels_pred = scores.argmax(dim=1).cpu()
        sim_mtx = torch.einsum('nd,cd->nc', [feas_sim.cpu(), feas_sim.cpu()])
        scores_k, idx_k = sim_mtx.topk(k=self.num_neighbor, dim=1)
        labels_samples = torch.zeros_like(idx_k)
        for s in range(self.num_neighbor):
            labels_samples[:, s] = labels_pred[idx_k[:, s]]

        true_mtx = labels_samples[:, 0:1] == labels_samples
        num_true = true_mtx.sum(dim=1)
        idx_true = num_true >= self.num_neighbor * self.ratio_confident
        print(num_true.min())
        idx_conf = scores.max(dim=1)[0].cpu() > self.score_th
        idx_true = idx_true * idx_conf
        idx_select = torch.where(idx_true > 0)[0]
        labels_select = labels_pred[idx_select]

        num_per_cluster = []
        idx_per_cluster = []
        label_per_cluster = []
        for c in range(self.num_cluster):
            idx_c = torch.where(labels_select == c)[0]
            idx_per_cluster.append(idx_select[idx_c])
            num_per_cluster.append(len(idx_c))
            label_per_cluster.append(labels_select[idx_c])

        idx_per_cluster_select = []
        label_per_cluster_select = []
        min_cluster = np.array(num_per_cluster).min()
        for c in range(self.num_cluster):
            idx_shuffle = np.arange(0, num_per_cluster[c])
            np.random.shuffle(idx_shuffle)
            idx_per_cluster_select.append(idx_per_cluster[c][idx_shuffle[0:min_cluster]])
            label_per_cluster_select.append(label_per_cluster[c][idx_shuffle[0:min_cluster]])

        idx_select = torch.cat(idx_per_cluster_select)
        labels_select = torch.cat(label_per_cluster_select)

        return idx_select, labels_select

    def compute_cluster_proto(self, feas_sim, scores):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        k = int(self.center_ratio * num_per_cluster)
        idx_max = idx_max[0:k, :]
        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)
        return centers

    def select_samples(self, feas_sim, scores, i):
        assert len(scores) == self.num_heads
        idx_select = []
        labels_select = []
        for h in range(self.num_heads):
            idx_select_h, labels_select_h = self.__getattr__("head_{}".format(h)).select_samples(feas_sim, scores[h], i)
            idx_select.append(idx_select_h)
            labels_select.append(labels_select_h)

        return idx_select, labels_select

    def forward(self, fea, **kwargs):
        cls_score = []
        if isinstance(fea, list):
            assert len(fea) == self.num_heads

        for h in range(self.num_heads):
            if isinstance(fea, list):
                cls_socre_h = self.__getattr__("head_{}".format(h)).forward(fea[h])
            else:
                cls_socre_h = self.__getattr__("head_{}".format(h)).forward(fea)

            cls_score.append(cls_socre_h)

        return cls_score

    def loss(self, x, target, **kwargs):
        assert len(x) == self.num_heads
        assert len(target) == self.num_heads

        loss = {}
        for h in range(self.num_heads):
            loss_h = self.__getattr__("head_{}".format(h)).loss(x[h], target[h])
            loss['head_{}'.format(h)] = loss_h

        return loss
