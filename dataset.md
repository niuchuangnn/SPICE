# Prepare datasets

The overall layout of dataset folder ```./datasets``` is as

    ├── datasets   
        ├── cifar10                   
        ├── cifar100                    
        ├── imagenet10_lmdb
        ├── imagenet10_npy            
        ├── imagenet_dog_lmdb
        ├── imagenet_dog_npy              
        ├── stl10
            ├──stl10_binary               
        ├── tiny_imagenet_lmdb
        ├── tiny_imagenet_val_lmdb
        ├── imagenet10_lmdb_meta_info.pkl
        ├── imagenet_dog_lmdb_meta_info.pkl
        ├── tiny_imagenet_lmdb_meta_info.pkl
        └── tiny_imagenet_val_lmdb_meta_info.pkl
        
These datasets can be downloaded as follows.
- [STL10](https://cs.stanford.edu/~acoates/stl10/)
```shell script
wget http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz
```
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
```shell script
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
```shell script
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
```
- The preprocessed versions of ImageNet10, ImageNetDog, and TinyImageNet can be downloaded [here](https://drive.google.com/file/d/154iP-M04oEB2TiieklPkVLU2mKO-2dKc/view?usp=sharing).
