import argparse
import os
import sys
sys.path.insert(0, './')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from spice.config import Config
from spice.data.build_dataset import build_dataset
from spice.model.build_model_sim import build_model_sim
from spice.utils.miscellaneous import mkdir, save_config
import numpy as np
from spice.utils.load_model_weights import load_model_weights


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    "--config-file",
    default="./configs/stl10/embedding.py",
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

    if cfg.gpu is not None:
        print("Use GPU: {}".format(cfg.gpu))

    # create model
    model_sim = build_model_sim(cfg.model_sim)
    print(model_sim)

    torch.cuda.set_device(cfg.gpu)
    model_sim = model_sim.cuda(cfg.gpu)

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
