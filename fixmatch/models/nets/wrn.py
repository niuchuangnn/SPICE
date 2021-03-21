import math
import torch
import torch.nn as nn
import torch.nn.functional as F

momentum=0.001

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bn_momentum=0.1, leaky_slope=0.0, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=bn_momentum)
        self.relu1 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=bn_momentum)
        self.relu2 = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, bn_momentum=0.1, leaky_slope=0.0, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, bn_momentum, leaky_slope, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, bn_momentum, leaky_slope, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, bn_momentum, leaky_slope, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, bn_momentum=0.1, leaky_slope=0.0, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, bn_momentum, leaky_slope, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, bn_momentum, leaky_slope, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, bn_momentum, leaky_slope, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=bn_momentum)
        self.relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        output = self.fc(out)
        if ood_test:
            return output, out
        else:
            return output
        
class build_WideResNet:
    def __init__(self, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0):
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
    
    def build(self, num_classes):
        return WideResNet(depth = self.depth,
                          num_classes = num_classes,
                          widen_factor = self.widen_factor,
                          bn_momentum = self.bn_momentum,
                          leaky_slope = self.leaky_slope,
                          dropRate = self.dropRate)
    
if __name__ == '__main__':
    wrn_builder = build_WideResNet(10, 2, 0.01, 0.1, 0.5)
    wrn = wrn_builder.build(10)
    print(wrn)
    
