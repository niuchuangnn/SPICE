# SPICE training

A step-by-step training tutorial for STL10 datast is as follows.

##### 1. Pretrain representation learning model, i.e., MoCo, assuming 4 GPUs available.
```shell script
python tools/train_moco.py
```
##### 2. Precompute embedding features
```shell script
python tools/pre_compute_embedding.py
```
##### 3. Train SPICE-Self
```shell script
python tools/train_self_v2.py
```
##### 4. Determine reliable images
```shell script
python tools/local_consistency.py
```

##### 5. Train SPICE-Semi, assuming 4 GPUs available.
```shell script
python ./tools/train_semi.py --unlabeled 1 --num_classes 10 --num_workers 4 --dist-url tcp://localhost:10001 --label_file ./results/stl10/eval/labels_reliable.npy --save_dir ./results/stl10/spice_semi --save_name semi --batch_size 64  --net WideResNet_stl10 --data_dir ./datasets/stl10 --dataset stl10
```
Note that ```--label_file``` and ```--save_name``` should be changed according to your generated reliable label file.

A step-by-step training tutorial for CIFAR-10 datast is as follows.

##### 1. Pretrain representation learning model, i.e., MoCo, assuming 4 GPUs available.
```shell script
python tools/train_moco.py --img_size 32 --moco-k 12800 --arch resnet18_cifar --save_folder ./results/cifar10/moco_res18_cls --resume ./results/cifar10/moco_res18_cls/checkpoint_last.pth.tar --data_type cifar10 --data ./datasets/cifar10 --all 0
```
##### 2. Precompute embedding features
```shell script
python tools/pre_compute_embedding.py --config-file configs/cifar10/embedding.py
```
##### 3. Train SPICE-Self
```shell script
python tools/train_self_v2.py --config-file ./configs/cifar10/spice_self.py --all 0
```
##### 4. Determine reliable images
```shell script
python tools/local_consistency.py --config-file ./configs/cifar10/eval.py --embedding ./results/cifar10/embedding/feas_moco_512_l2.npy
```

##### 5. Train SPICE-Semi, assuming 4 GPUs available.
```shell script
python ./tools/train_semi.py --all 0 --num_classes 10 --num_workers 16 --dist-url tcp://localhost:10001 --label_file ./results/cifar10/eval/labels_reliable.npy --save_dir ./results/cifar10/spice_semi --save_name semi --batch_size 64  --net resnet18_cifar --data_dir ./datasets/cifar10 --dataset cifar10
```
