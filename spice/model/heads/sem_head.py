import torch
import torch.nn as nn
from ..feature_modules.build_feature_module import build_feature_module

import numpy as np
import torch.nn.functional as F


class SemHead(nn.Module):
    def __init__(self, classifier, feature_conv=None, num_cluster=10, center_ratio=0.5,
                 iter_start=0, iter_up=-1, iter_down=-1, iter_end=0, ratio_start=0.5, ratio_end=0.95, loss_weight=None,
                 fea_fc=False, T=1, sim_ratio=1, sim_center_ratio=0.9, epoch_merge=5, entropy=False):

        super(SemHead, self).__init__()
        if loss_weight is None:
            loss_weight = dict(loss_cls=1, loss_ent=0)
        self.loss_weight = loss_weight
        self.classifier = build_feature_module(classifier)
        self.feature_conv = None
        if feature_conv:
            self.feature_conv = build_feature_module(feature_conv)

        self.num_cluster = num_cluster
        self.loss_fn_cls = nn.CrossEntropyLoss()
        self.iter_start = iter_start
        self.iter_end = iter_end
        self.ratio_start = ratio_start
        self.ratio_end = ratio_end
        self.center_ratio = center_ratio
        self.ave_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fea_fc = fea_fc
        self.T = T
        self.sim_ratio = sim_ratio
        self.iter_up = iter_up
        self.iter_down = iter_down
        self.sim_center_ratio = sim_center_ratio
        self.epoch_merge = epoch_merge

        self.entropy = entropy
        self.EPS = 1e-5

    def compute_ratio_selection_old(self, i):
        if self.ratio_end == self.ratio_start:
            return self.ratio_start
        elif self.iter_start < i <= self.iter_end:
            r = (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_start) * (i - self.iter_start) + self.ratio_start
            return r
        else:
            return self.ratio_start

    def compute_ratio_selection(self, i):
        if self.ratio_end == self.ratio_start:
            return self.ratio_start
        elif self.iter_up != -1 and self.iter_down != -1:
            if i < self.iter_start:
                return self.ratio_start
            elif self.iter_start <= i < self.iter_up:
                r = (self.ratio_end - self.ratio_start) / (self.iter_up - self.iter_start) * (i - self.iter_start) + self.ratio_start
                return r
            elif self.iter_up <= i < self.iter_down:
                return self.ratio_end
            elif self.iter_down <= i < self.iter_end:
                r = self.ratio_end - (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_down) * (i - self.iter_down)
                return r
            else:
                return self.ratio_start
        else:
            if self.iter_start < i <= self.iter_end:
                r = (self.ratio_end - self.ratio_start) / (self.iter_end - self.iter_start) * (i - self.iter_start) + self.ratio_start
                return r
            else:
                return self.ratio_start

    def select_samples_cpu(self, feas_sim, scores, i):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        print(k)
        idx_max = idx_max[0:k, :]

        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0))

        select_idx_all = []
        select_labels_all = []
        num_per_cluster = feas_sim.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        num_select_c = int(num_per_cluster * ratio_select)
        for c in range(self.num_cluster):
            center_c = centers[c]
            dis = np.dot(feas_sim, center_c.T).squeeze()
            idx_s = np.argsort(dis)[::-1]
            idx_select = idx_s[0:num_select_c]

            select_idx_all = select_idx_all + list(idx_select)
            select_labels_all = select_labels_all + [c] * len(idx_select)

        select_idx_all = np.array(select_idx_all)
        select_labels_all = np.array(select_labels_all)

        return select_idx_all, select_labels_all

    def select_samples(self, feas_sim, scores, i):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        # print(ratio_select)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        # print(k, len(idx_max))
        idx_max = idx_max[0:k, :]

        centers = []
        for c in range(self.num_cluster):
            centers.append(feas_sim[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)

        num_select_c = int(num_per_cluster * ratio_select)

        dis = torch.einsum('cd,nd->cn', [centers, feas_sim])
        idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
        labels_select = torch.arange(0, self.num_cluster).unsqueeze(dim=1).repeat(1, num_select_c).flatten()

        return idx_select, labels_select

    def select_samples_v2(self, feas_sim, scores, i):

        _, idx_max = torch.sort(scores, dim=0, descending=True)
        idx_max = idx_max.cpu()
        num_per_cluster = idx_max.shape[0] // self.num_cluster
        ratio_select = self.compute_ratio_selection(i)
        # print(ratio_select)
        k = int(self.center_ratio * num_per_cluster * ratio_select)
        # print(k, len(idx_max))

        idx_center_exist = torch.zeros_like(idx_max[:, 0], dtype=torch.bool)

        centers = []
        for c in range(self.num_cluster):
            idx_c = idx_max[:, c]
            if c == 0:
                idx_c_select = idx_c[0:k]
            else:
                idx_c_available = ~idx_center_exist[idx_c]
                idx_c_select = idx_c[idx_c_available][0:k]

            idx_center_exist[idx_c_select] = True

            centers.append(feas_sim[idx_c_select, :].mean(axis=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)

        num_select_c = int(num_per_cluster * ratio_select)

        dis = torch.einsum('cd,nd->cn', [centers, feas_sim])
        # idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
        idx_sort = torch.argsort(dis, dim=1, descending=True)
        idx_label_exist = torch.zeros_like(idx_sort[0, :], dtype=torch.bool)
        labels_select_all = []
        idx_select_all = []
        for c in range(self.num_cluster):
            idx_c = idx_sort[c, :]
            if c == 0:
                idx_c_select = idx_sort[0, 0:num_select_c]
            else:
                idx_c_available = ~idx_label_exist[idx_c]
                idx_c_select = idx_c[idx_c_available][0:num_select_c]

            idx_label_exist[idx_c_select] = True
            idx_select_all.append(idx_c_select)
            labels_select_all.append(torch.zeros_like(idx_c_select)+c)

        idx_select_all = torch.cat(idx_select_all)
        labels_select_all = torch.cat(labels_select_all)
        print(len(set(idx_select_all.cpu().numpy())))

        return idx_select_all, labels_select_all

    def forward(self, fea, **kwargs):

        if self.feature_conv is not None:
            fea_conv = self.feature_conv(fea)
        else:
            fea_conv = fea

        if not self.fea_fc:
            feature = self.ave_pooling(fea_conv)
            feature = feature.flatten(start_dim=1)
        else:
            feature = fea_conv

        cls_score = self.classifier(feature)

        cls_score = cls_score / self.T

        return cls_score

    def loss(self, x, target, **kwargs):
        cls_socre = self.forward(x)
        loss = self.loss_fn_cls(cls_socre, target) * self.loss_weight["loss_cls"]

        if self.entropy:
            prob_mean = cls_socre.mean(dim=0)
            prob_mean[(prob_mean < self.EPS).data] = self.EPS
            loss_ent = (prob_mean * torch.log(prob_mean)).sum()
            loss = loss + loss_ent * self.loss_weight["loss_ent"]

        return loss
