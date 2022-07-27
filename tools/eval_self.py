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
from matplotlib.pyplot import imsave
from PIL import Image
import matplotlib.pyplot as plt
import PIL


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
    "--weight",
    default="./model_zoo/self_model_stl10.pth.tar",
    metavar="FILE",
    help="path to weight file",
    type=str,
)
parser.add_argument(
    "--all",
    default=1,
    type=int,
)
parser.add_argument(
    "--proto",
    default=0,
    type=int,
)
parser.add_argument(
    "--embedding",
    default="./results/stl10/embedding/feas_moco_512_l2.npy",
    type=str,
)


def main():
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    cfg.model.pretrained = args.weight
    cfg.proto = args.proto
    cfg.embedding = args.embedding
    cfg.all = args.all
    if cfg.all:
        cfg.data_test.split = "train+test"
        cfg.data_test.all = True
    else:
        cfg.data_test.split = "test"
        cfg.data_test.all = False

    output_dir = cfg.results.output_dir
    if output_dir:
        mkdir(output_dir)
        if cfg.proto:
            mkdir("{}/proto".format(output_dir))

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
    for k in list(state_dict.keys()):
        # Initialize the feature module with encoder_q of moco.
        if k.startswith('module.'):
            # remove prefix
            # state_dict["module.{}".format(k[len('module.encoder_q.'):])] = state_dict[k]
            state_dict["{}".format(k[len('module.'):])] = state_dict[k]

        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict)

    # Load similarity model
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

    pred_labels = torch.cat(pred_labels).long().cpu().numpy()
    scores = torch.cat(scores_all).cpu()

    try:
        acc = calculate_acc(pred_labels, gt_labels)
    except:
        acc = -1

    nmi = calculate_nmi(pred_labels, gt_labels)
    ari = calculate_ari(pred_labels, gt_labels)

    print("ACC: {}, NMI: {}, ARI: {}".format(acc, nmi, ari))

    if cfg.proto:
        data = val_loader.dataset.data
        feas_sim = np.load(cfg.embedding)
        feas_sim = torch.from_numpy(feas_sim)
        centers = model(feas_sim=feas_sim, scores=scores, forward_type="proto")

        sim_all = torch.einsum('nd,cd->nc', [feas_sim.cpu(), centers.cpu()])

        _, top_10 = torch.topk(sim_all, 10, 0)

        imgs = []
        for c in range(cfg.num_cluster):
            idx_c = top_10[:, c]
            img_c = data[idx_c, ...]
            imgs.append(img_c)
            for ii in range(10):
                img_c_ii = img_c[ii, ...].transpose([1, 2, 0])
                imsave('{}/proto/{}_{}.png'.format(cfg.results.output_dir, c, ii), img_c_ii)

        for c in range(cfg.num_cluster):
            dataset_val.data = imgs[c]
            for i in range(len(dataset_val)):
                img, _, labels, idx = dataset_val[i]
                img = torch.unsqueeze(img, dim=0).to(cfg.gpu, non_blocking=True)
                with torch.no_grad():
                    fea_conv = model(img, forward_type="feature_only")
                fea_conv = fea_conv.reshape(512, 49)
                center = centers[c:c+1, :]
                sim_map = torch.einsum('nd,dm->nm', [center.cpu(), fea_conv.cpu()])
                sim_map = sim_map.reshape([7, 7])
                sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
                sim_map = sim_map.cpu().numpy()

                img_c_ii = imgs[c][i, ...].transpose([1, 2, 0])

                sim_map = Image.fromarray(np.uint8(sim_map * 255))
                sim_map = sim_map.resize((img_c_ii.shape[1], img_c_ii.shape[0]), resample=PIL.Image.BILINEAR)
                sim_map = np.asarray(sim_map)

                att_mask = np.zeros_like(img_c_ii)
                att_mask[:, :, 0] = sim_map

                cmap = plt.get_cmap('jet')
                attMap = sim_map
                attMapV = cmap(attMap)
                attMapV = np.delete(attMapV, 3, 2) * 255

                attMap = 0.6 * img_c_ii + 0.4 * attMapV
                attMap = attMap.astype(np.uint8)
                imsave('{}/proto/{}_{}_att.png'.format(cfg.results.output_dir, c, i), attMap)


if __name__ == '__main__':
    main()
