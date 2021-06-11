# SPICE: Semantic Pseudo-labeling for Image Clustering
By [Chuang Niu](https://scholar.google.com/citations?user=aoud5NgAAAAJ&hl) and [Ge Wang](https://www.linkedin.com/in/ge-wang-axis/)

This is a Pytorch implementation of the [paper](https://arxiv.org/pdf/2103.09382.pdf). (**In updating**)


<tr>
<p align="center"> <img height="360" src="./figures/framework.png"></p>
</tr>


<tr>
<td><img  height="190" src="./figures/proto-local.png"></td>
</tr>

- **SOTA on 5 benchmarks. Please refer to [Papers With Code](https://paperswithcode.com/paper/spice-semantic-pseudo-labeling-for-image) for [Image Clustering](https://paperswithcode.com/task/image-clustering)**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spice-semantic-pseudo-labeling-for-image/image-clustering-on-stl-10)](https://paperswithcode.com/sota/image-clustering-on-stl-10?p=spice-semantic-pseudo-labeling-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spice-semantic-pseudo-labeling-for-image/image-clustering-on-cifar-10)](https://paperswithcode.com/sota/image-clustering-on-cifar-10?p=spice-semantic-pseudo-labeling-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spice-semantic-pseudo-labeling-for-image/image-clustering-on-cifar-100)](https://paperswithcode.com/sota/image-clustering-on-cifar-100?p=spice-semantic-pseudo-labeling-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spice-semantic-pseudo-labeling-for-image/image-clustering-on-imagenet-10)](https://paperswithcode.com/sota/image-clustering-on-imagenet-10?p=spice-semantic-pseudo-labeling-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/spice-semantic-pseudo-labeling-for-image/image-clustering-on-tiny-imagenet)](https://paperswithcode.com/sota/image-clustering-on-tiny-imagenet?p=spice-semantic-pseudo-labeling-for-image)


## Installation
Please refer to [requirement.txt](./requirements.txt) for all required packages.
Assuming [Anaconda](https://www.anaconda.com/) with python 3.7, a step-by-step example for installing this project is as follows:

```shell script
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge addict tensorboard python-lmdb
conda install matplotlib scipy scikit-learn pillow
```

Then, clone this repo
```shell script
git clone https://github.com/niuchuangnn/SPICE.git
cd SPICE
```

## Data
Prepare datasets of interest as described in [dataset.md](./dataset.md).

## Training
Read the [training tutorial](./train.md) for details.

## Evaluation
Evaluation of SPICE-Self:
```shell script
python tools/eval_self.py --config-file configs/stl10/eval.py --weight PATH/TO/MODEL --all 1
```
Evaluation of SPICE-Semi:
```shell script
python tools/eval_semi.py --load_path PATH/TO/MODEL --net WideResNet --widen_factor 2 --data_dir PATH/TO/DATA --dataset cifar10 --all 1 
```
Read the [evaluation tutorial](./evaluation.md) for more descriptions about the evaluation and the visualization of learned clusters.

## Model Zoo
All trained models in our paper are available as follows.

| Dataset          | Version           | ACC                   |  NMI            |  ARI      |Model link |
|------------------|-------------------|---------------------- |-----------------|-----------|--------------|
| STL10            |  SPICE-Self       | 91.0                  | 82.0            | 81.5      |[Model](https://drive.google.com/file/d/1rwGQgQaDdrWOk7zVhROEJjZ87eXRu0sw/view?usp=sharing)  |
|                  |  SPICE            | 93.8                  | 87.2            | 87.0      |[Model](https://drive.google.com/file/d/1czgfXh3bJPgU19-HQjb019cU2mjQccFY/view?usp=sharing)  |
|                  |  SPICE-Self*      | 89.9                  | 80.9            | 79.7      |[Model](https://drive.google.com/file/d/18bhdRcwXxQHfzNlm1ZFjQnIhCHcQKb_Z/view?usp=sharing)  |
|                  |  SPICE*           | 92.9                  | 86.0            | 85.3      |[Model](https://drive.google.com/file/d/1IFhsS6I0GjEO33TcUuZnHbV3zs6SzAqe/view?usp=sharing)  |
| CIFAR10          |  SPICE-Self       | 83.8                  | 73.4            | 70.5      |[Model](https://drive.google.com/file/d/1Qvti7K8UTVKDsa34WReJKPyeBElfg2Xd/view?usp=sharing)  |
|                  |  SPICE            | 92.6                  | 86.5            | 85.2      |[Model](https://drive.google.com/file/d/1rpCgghJNdlecguGBKsXv97Q9uHN0fSCA/view?usp=sharing) |
|                  |  SPICE-Self*      | 84.9                  | 74.5            | 71.8      |[Model](https://drive.google.com/file/d/1ILexWaM2zR00IjR0H567umQACfFzIJsZ/view?usp=sharing)  |
|                  |  SPICE*           | 91.7                  | 85.8            | 83.6      |[Model](https://drive.google.com/file/d/1QriNjz-08ca8uH9X-lf84WNdmRVSY4OW/view?usp=sharing)  |
| CIFAR100         |  SPICE-Self       | 46.8                  | 44.8            | 29.4      |[Model](https://drive.google.com/file/d/1XjFNz4Xf-nMO5AbloUWFeULeW3_6LooS/view?usp=sharing)  |
|                  |  SPICE            | 53.8                  | 56.7            | 38.7      |[Model](https://drive.google.com/file/d/1b2OakQjRu8vVXcLdRzFlOYoA71GzBDa0/view?usp=sharing)  |
|                  |  SPICE-Self*      | 48.0                  | 45.0            | 30.8      |[Model](https://drive.google.com/file/d/1FLNdRw2kewvH06ROrsupcw8p6nMraxQ1/view?usp=sharing)  |
|                  |  SPICE*           | 58.4                  | 58.3            | 42.2      |[Model](https://drive.google.com/file/d/1u8ajijNtLhDcypFRtJcCV1Njm8iMHkvP/view?usp=sharing)  |
| ImageNet-10      |  SPICE-Self       | 96.9                  | 92.7            | 93.3      |[Model](https://drive.google.com/file/d/1C2ERpAVAnNgtX7OQON4od4_BBAubSMPr/view?usp=sharing)  |
|                  |  SPICE            | 96.7                  | 91.7            | 92.9      |[Model](https://drive.google.com/file/d/18AsOvwE0ElgHipcSKlwUad8-p3mv6EEm/view?usp=sharing)  |
| ImageNet-Dog     |  SPICE-Self       | 54.6                  | 49.8            | 36.2      |[Model](https://drive.google.com/file/d/1pLcfEydw4L7yy_xlhgDq2nVImHooWdkQ/view?usp=sharing)  |
|                  |  SPICE            | 55.4                  | 50.4            | 34.3      |[Model](https://drive.google.com/file/d/1-qqckC8N9_zcIyI8NjB51qApDalFoXti/view?usp=sharing)  |
| TinyImageNet     |  SPICE-Self       | 30.5                  | 44.9            | 16.3      |[Model](https://drive.google.com/file/d/1JnmptRFP5rNM61JI9ehwXLyfWtE3jS8l/view?usp=sharing)  |
|                  |  SPICE-Self*      | 29.2                  | 52.5            | 14.5      |[Model](https://drive.google.com/file/d/1JnmptRFP5rNM61JI9ehwXLyfWtE3jS8l/view?usp=sharing)  |

More models based on ResNet18 for both SPICE-Self* and SPICE-Semi*.

| Dataset          | Version           | ACC                   |  NMI            |  ARI      |Model link |
|------------------|-------------------|---------------------- |-----------------|-----------|--------------|
| STL10            |  SPICE-Self*      | 86.2                  | 75.6            | 73.2      |[Model](https://drive.google.com/file/d/1fZ7RJOAUB5dFkmVGxuRTCe0sT5DoltyN/view?usp=sharing)  |
|                  |  SPICE*           | 92.0                  | 85.2            | 83.6      |[Model](https://drive.google.com/file/d/1N8NhMGPeu_S9hiuLSKfdfxOHbhvE1pOu/view?usp=sharing)  |
| CIFAR10          |  SPICE-Self*      | 84.5                  | 73.9            | 70.9      |[Model](https://drive.google.com/file/d/1J3gqkLIK5wPC3Vuw4zCjU0IpXCAxqsyx/view?usp=sharing)  |
|                  |  SPICE*           | 91.8                  | 85.0            | 83.6      |[Model](https://drive.google.com/file/d/14qRE6lmzPOZPYSso-xZPfdVggWuVa7dB/view?usp=sharing) |
| CIFAR100         |  SPICE-Self*      | 46.8                  | 45.7            | 32.1      |[Model](https://drive.google.com/file/d/11I__pO5n-OFRuh6OQ31IFdNBPpUtV9sv/view?usp=sharing)  |
|                  |  SPICE*           | 53.5                  | 56.5            | 40.4      |[Model](https://drive.google.com/file/d/1K_3WRqJZA7GqXBhSFb2RLMQOSfRmgohe/view?usp=sharing)  |


## Acknowledgement for reference repos
- [GATCluster](https://github.com/niuchuangnn/GATCluster)
- [MOCO](https://github.com/facebookresearch/moco)
- [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification)
- [FixMatch](https://github.com/LeeDoYup/FixMatch-pytorch)
- [IIC](https://github.com/xu-ji/IIC)

## Citation

```shell
@misc{niu2021spice,
      title={SPICE: Semantic Pseudo-labeling for Image Clustering}, 
      author={Chuang Niu and Ge Wang},
      year={2021},
      eprint={2103.09382},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```