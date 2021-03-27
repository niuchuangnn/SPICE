# SPICE training

A step-by-step training tutorial for STL10 datast is as follows.

##### 1. Pretrain representation learning model, i.e., MoCo, assuming 4 GPUs available.
```shell script
python tools/train_moco.py
```
##### 2. Precompute embedding features
```shell script
python tools/pre_compute_ebmedding.py
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
python ./tools/train_semi.py --unlabeled 1 --num_classes 10 --num_workers 4 --dist-url tcp://localhost:10001 --label_file ./results/stl10/eval/labels_reliable_0.983136_6760.npy --save_dir ./results/stl10/spice_semi --save_name 098_6760 --batch_size 64  --net WideResNet_stl10 --data_dir ./datasets/stl10 --dataset stl10
```
Note that ```--label_file``` and ```--save_name``` should be changed according to your generated reliable label file.

TODO: More training descriptions on other datasets will be added, and some training steps will be merged.