model_name = "eval"
weight = './model_zoo/self_model_cifar10.pth.tar'
model_type = "clusterresnet"
# model_type = "resnet18_cifar"
device_id = 0
num_cluster = 10
batch_size = 1000
fea_dim = 512
center_ratio = 0.5
world_size = 1
workers = 4
rank = 0
dist_url = 'tcp://localhost:10001'
dist_backend = "nccl"
seed = None
gpu = None
multiprocessing_distributed = True

data_test = dict(
    type="cifar10",
    root_folder="./datasets/cifar10",
    embedding=None,
    train=True,
    all=True,
    shuffle=False,
    ims_per_batch=50,
    aspect_ratio_grouping=False,
    show=False,
    trans1=dict(
        aug_type="test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),
    trans2=dict(
        aug_type="test",
        normalize=dict(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
    ),

)

model = dict(
    feature=dict(
        type=model_type,
        num_classes=num_cluster,
        in_channels=3,
        in_size=32,
        batchnorm_track=True,
        test=False,
        feature_only=True
    ),

    head=dict(type="sem_multi",
              multi_heads=[dict(classifier=dict(type="mlp", num_neurons=[fea_dim, fea_dim, num_cluster], last_activation="softmax"),
                                feature_conv=None,
                                num_cluster=num_cluster,
                                ratio_start=1,
                                ratio_end=1,
                                center_ratio=center_ratio,
                                )]*1,
              ratio_confident=0.99,
              num_neighbor=100,
              ),
    model_type="moco_select",
    pretrained=weight,
    head_id=6,
    freeze_conv=True,
)

results = dict(
    output_dir="./results/cifar10/{}".format(model_name),
)