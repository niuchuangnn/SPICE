# Evaluation of SPICE

>An example on STL10:

SPICE-Self:
```shell script
 python tools/eval_self.py --config-file configs/stl10/eval.py --weight model_zoo/self_model_stl10.pth.tar --all 1
```
SPICE
```shell script
python tools/eval_semi.py --load_path ./model_zoo/model_stl10.pth --net WideResNet_stl10 --widen_factor 2 --data_dir ./datasets/stl10 --dataset stl10 --all 1 --num_classes 10
```
SPICE-Self*:
```shell script
 python tools/eval_self.py --config-file configs/stl10/eval.py --weight model_zoo/self_model_stl10_cls.pth.tar --all 0 
```
SPICE*
```shell script
python tools/eval_semi.py --load_path ./model_zoo/model_stl10.pth --net WideResNet_stl10 --widen_factor 2 --data_dir ./datasets/stl10 --dataset stl10 --all 0 --num_classes 10
```

- Visualization of learned cluster semantics
```shell script
python tools/eval_self.py --config-file configs/stl10/eval.py --weight model_zoo/self_model_stl10.pth.tar --all 1 --proto 1 ----embedding ./results/stl10/embedding/feas_moco_512_l2.npy
```
Then, the visualization results will be saved in ```./results/stl10/eval/proto/```, including both the prototype examples and the corresponding localization results. 