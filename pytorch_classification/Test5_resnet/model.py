import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride = 1, downsample = None, **kwargs):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=3, padding=1, stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 groups=1,
                 width_per_group=64):
        super(Bottleneck, self).__init__()
        self.out_channels = int(out_channels*(width_per_group / 64.)*groups)
        self.downsample=downsample
        self.conv1 = nn.Conv2d(in_channels, self.out_channels,kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,
                               kernel_size=3,stride=stride,padding=1,groups=groups,bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = nn.Conv2d(self.out_channels, out_channels*self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    in_channel = 64
    def __init__(self,
                 block,
                 blocks_num:list,
                 num_class = 1000,
                 include_top = True,
                 groups=1,
                 width_per_groups=64):
        super(ResNet, self).__init__()
        self.groups = groups
        self.width_per_groups = width_per_groups
        self.in_channel = 64
        self.include_top = include_top
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layers(block, 64, blocks_num[0], 1)
        self.layer2 = self._make_layers(block, 128, blocks_num[1], 2)
        self.layer3 = self._make_layers(block, 256, blocks_num[2], 2)
        self.layer4 = self._make_layers(block, 512, blocks_num[3], 2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(in_features=512*block.expansion,out_features=num_class)

    def _make_layers(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layers=[]
        layers.append(block(
            self.in_channel,channel,
            stride = stride,
            downsample = downsample,
            groups = self.groups,
            width_per_group=self.width_per_groups
        ))
        self.in_channel = channel*block.expansion
        for i in range(1,block_num):
            layers.append(
                block(
                    self.in_channel, channel, groups = self.groups, width_per_group = self.width_per_groups
                ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

def resnet34(num_class = 1000, include_top = True):
    return ResNet(BasicBlock,[3,4,6,3],num_class,include_top)

def resnet50(num_class = 1000, include_top = True):
    return ResNet(Bottleneck,[3,4,6,3],num_class,include_top)

def resnet101(num_class = 1000, include_top =True):
    return ResNet(Bottleneck,[3,4,23,3],num_class,include_top)

def resnext50_32x4d(num_class = 1000, include_top = True):
    return ResNet(Bottleneck,[3,4,6,3],num_class,include_top,32,4)

def resnext101_32x8d(num_class = 1000, include_top = True):
    return ResNet(Bottleneck,[3,4,23,3],num_class,include_top,32,8)