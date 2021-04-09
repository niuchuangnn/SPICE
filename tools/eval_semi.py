from __future__ import print_function, division
import os

import torch
import sys
sys.path.insert(0, './')

from fixmatch.utils import net_builder
from fixmatch.datasets.ssl_dataset_robust import SSL_Dataset
from fixmatch.datasets.data_utils import get_data_loader
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


def calculate_acc(ypred, y, return_idx=False):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.

    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.

    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    row, col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    if return_idx:
        return 1.0 * count / len(y), row, col
    else:
        return 1.0 * count / len(y)


def calculate_nmi(predict_labels, true_labels):
    # NMI
    nmi = metrics.normalized_mutual_info_score(true_labels, predict_labels, average_method='geometric')
    return nmi


def calculate_ari(predict_labels, true_labels):
    # ARI
    ari = metrics.adjusted_rand_score(true_labels, predict_labels)
    return ari


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./model_zoo/model_stl10.pth')
    parser.add_argument('--scores_path', type=str, default=None)
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='./datasets/cifar10')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--label_file', type=str, default=None)
    parser.add_argument('--all', type=int, default=0)
    parser.add_argument('--unlabeled', type=bool, default=False)
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['train_model'] if args.use_train_model else checkpoint['eval_model']

    for k in list(load_model.keys()):

        # Initialize the feature module with encoder_q of moco.
        if k.startswith('model.'):
            # remove prefix
            load_model[k[len('model.'):]] = load_model[k]

            del load_model[k]
            # print(k)

    if args.net in ['WideResNet', 'WideResNet_stl10', 'WideResNet_tiny', 'resnet18', 'resnet18_cifar']:
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'depth': args.depth,
                                    'widen_factor': args.widen_factor,
                                    'leaky_slope': args.leaky_slope,
                                    'dropRate': args.dropout})
    elif args.net == 'ClusterResNet':
        _net_builder = net_builder(args.net,
                                   args.net_from_name,
                                   {'input_size': args.input_size})
    else:
        raise TypeError
    
    net = _net_builder(num_classes=args.num_classes)
    net.load_state_dict(load_model)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    _eval_dset = SSL_Dataset(name=args.dataset, train=False, data_dir=args.data_dir, label_file=None, all=args.all, unlabeled=False)
    # print(args.all)

    eval_dset = _eval_dset.get_dset()
    print(len(eval_dset))
    
    eval_loader = get_data_loader(eval_dset,
                                  args.batch_size, 
                                  num_workers=1)
 
    acc = 0.0
    labels_pred = []
    labels_gt = []
    scores = []
    with torch.no_grad():
        for image, target, _ in eval_loader:
            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)

            scores.append(logit.cpu().numpy())

            labels_pred.append(torch.max(logit, dim=-1)[1].cpu().numpy())
            labels_gt.append(target.cpu().numpy())

    scores = np.concatenate(scores, axis=0)
    labels_pred = np.concatenate(labels_pred, axis=0)
    labels_gt = np.concatenate(labels_gt, axis=0)
    try:
        acc = calculate_acc(labels_pred, labels_gt)
    except:
        acc = -1

    nmi = calculate_nmi(labels_pred, labels_gt)
    ari = calculate_ari(labels_pred, labels_gt)
            # acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
    
    print(f"Test Accuracy: {acc}, NMI: {nmi}, ARI: {ari}")
    # print(len(eval_dset))

    if args.scores_path is not None:
        np.save(args.scores_path, scores)

