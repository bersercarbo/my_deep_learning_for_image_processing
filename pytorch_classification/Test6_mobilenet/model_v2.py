import torch.nn as nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class Conv2dBNReLU(nn.Sequential):
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,
                      stride=stride,groups=groups,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self,in_channels, out_channels, stride, exposion):
        super(InvertedResidual, self).__init__()
        self.shortcut = in_channels == out_channels and stride == 1
        mid_channels = in_channels * exposion
        layers = []
        if exposion != 1:
            layers.append(Conv2dBNReLU(in_channels,mid_channels,kernel_size=1))

        layers.extend([
            Conv2dBNReLU(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=mid_channels),
            nn.Conv2d(mid_channels, out_channels,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self,x):
        if self.shortcut :
            return x + self.conv(x)
        return self.conv(x)
    
class MobileNetV2(nn.Module):
    def __init__(self, num_class=1000, alph=1.0, divisor=8):
        super(MobileNetV2, self).__init__()
        self.in_channels = _make_divisible(32*alph, divisor)
        self.last_channels = _make_divisible(1280*alph, divisor)
        block=InvertedResidual
        features = []
        features.append(Conv2dBNReLU(3,self.in_channels,kernel_size=3,stride=2))
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        for t,c,n,s in inverted_residual_setting:
            out_channels = _make_divisible(c*alph, divisor)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(in_channels=self.in_channels,out_channels=out_channels,stride=stride,exposion=t))
                self.in_channels = out_channels

        features.append(Conv2dBNReLU(in_channels=self.in_channels,out_channels=self.last_channels,kernel_size=1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.last_channels, out_features=num_class)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x