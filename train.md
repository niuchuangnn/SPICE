# SPICE training

A step-by-step training tutorial for STL10 datast is as follows.

##### 1. Pretrain representation learning model, i.e., MoCo, assuming 4 GPUs available.
```shell script
python tools/train_moco_stl10.py
```
##### 2. Precompute embedding features
```shell script
python tools/pre_compute_ebmedding.py
```
##### 3. Train SPICE-Self
```shell script
python tools/train_self.py
```
##### 4. Determine reliable images
```shell script
python tools/local_consistency.py
```

##### 5. Train SPICE-Semi
```shell script
python tools/train_semi.py
```

TODO: More training descriptions on other datasets will be added, and some training steps will be merged.