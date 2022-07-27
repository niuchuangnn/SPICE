import argparse
import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from spice.config import Config
from spice.data.build_dataset import build_dataset
from spice.model.sim2sem import Sim2Sem
from spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from spice.utils.evaluation import calculate_acc, calculate_nmi, calculate_ari


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

    cfg.embedding = args.embedding

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)

    output_config_path = os.path.join(output_dir, 'config.py')
    save_config(cfg, output_config_path)

    if cfg.gpu is not None:
        print("Use GPU: {}".format(cfg.gpu))

    # create model
    model = Sim2Sem(**cfg.model)
    print(model)

    torch.cuda.set_device(cfg.gpu)
    model = model.cuda(cfg.gpu)

    state_dict = torch.load(cfg.model.pretrained)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    for k in list(state_dict.keys()):
        # Initialize the feature module with encoder_q of moco.
        if k.startswith('module.'):
            # remove prefix
            # state_dict["module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            state_dict["{}".format(k[len('module.'):])] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]

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

    for _, (images, _, labels, idx) in enumerate(val_loader):
        images = images.to(cfg.gpu, non_blocking=True)
        with torch.no_grad():
            scores = model(images, forward_type="sem")

        assert len(scores) == num_heads

        pred_idx = scores[0].argmax(dim=1)
        pred_labels.append(pred_idx)
        scores_all.append(scores[0])

        gt_labels.append(labels)

    gt_labels = torch.cat(gt_labels).long().cpu().numpy()
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

    np.save("{}/labels_reliable.npy".format(cfg.results.output_dir), labels_correct)


if __name__ == '__main__':
    main()
