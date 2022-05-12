import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

cfgs = {
    #   size     224            112       56                        28                      14                         7
    'vgg11': [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGnet(nn.Module):
    def __init__(self, features, num_class = 1000, init_weights = False):
        super(VGGnet, self).__init__()
        self.features = features
        self.classifiers = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096,num_class)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,start_dim=1)
        x = self.classifiers(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_features(cfg:list):
    layers = []
    in_channels = 3
    for i in cfg:
        if i == "M":
            layers += [nn.MaxPool2d(2, 2)]
        else:
            layers += [nn.Conv2d(in_channels, i, kernel_size=3, padding=1)]
            layers += [nn.ReLU(True)]
            in_channels = i
    return nn.Sequential(*layers)

def vgg(model_name = "vgg16", **kwargs):
    assert model_name in cfgs, "Warning:model number {} not in cfgs dict!".format()\
                               +"model number have vgg11, vgg13, vgg 16, vgg19."
    cfg = cfgs[model_name]
    model = VGGnet(features = make_features(cfg), **kwargs)
    return model

# print(vgg("vgg16"))