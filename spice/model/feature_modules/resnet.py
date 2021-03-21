import torch.nn as nn
from torchvision.models import resnet


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None,
               track_running_stats=None):
    super(BasicBlock, self).__init__()

    assert (track_running_stats is not None)

    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, track_running_stats=track_running_stats)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNetTrunk(nn.Module):
  def __init__(self):
    super(ResNetTrunk, self).__init__()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion,
                       track_running_stats=self.batchnorm_track),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample,
                        track_running_stats=self.batchnorm_track))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(
        block(self.inplanes, planes, track_running_stats=self.batchnorm_track))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
  def __init__(self):
    super(ResNet, self).__init__()

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        assert (m.track_running_stats == self.batchnorm_track)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class ResNet5gTrunk(ResNetTrunk):
  def __init__(self, in_channels, batchnorm_track=True, fea_dim=10):
    super(ResNet5gTrunk, self).__init__()

    self.batchnorm_track = batchnorm_track

    block = BasicBlock
    layers = [3, 4, 6, 3]

    in_channels = in_channels
    self.inplanes = 64
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                           padding=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    self.layer5 = nn.Conv2d(512, fea_dim, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn5 = nn.BatchNorm2d(fea_dim, track_running_stats=self.batchnorm_track)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    x = self.layer4(x)

    x = self.layer5(x)
    x = self.bn5(x)

    return x
