import torch.nn as nn
import torch
import torch.nn.functional as F

class GoogLeNet(nn.Module):
    def __init__(self, num_class=1000, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.conv1 = BasicConv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True)
        self.conv2 = nn.Sequential(
            BasicConv2d(64,64,kernel_size=1),
            BasicConv2d(64,192,kernel_size=3,padding=1)
        )
        self.inception3 = nn.Sequential(
            Inception(192, 64,  96,  128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64)
        )
        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4bcd = nn.Sequential(
            Inception(512,160,112,224,24,64,64),
            Inception(512,128,128,256,24,64,64),
            Inception(512,112,144,288,32,64,64)
        )
        self.inception4e = Inception(528,256,160,320,32,128,128)
        self.inception5 = nn.Sequential(
            Inception(832,256,160,320,32,128,128),
            Inception(832,384,192,384,48,128,128)
        )
        self.avgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(1024, num_class)
        )
        self.aux14a = InceptionAux(512, num_class)
        self.aux24d = InceptionAux(528, num_class)
        if init_weights==True:
            self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        if self.training == True:
            aux1 = self.aux14a(x)
        x = self.inception4bcd(x)
        if self.training == True:
            aux2 = self.aux24d(x)
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5(x)
        x = self.avgPool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        if self.training == True:
            return x,aux1,aux2
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)

class BasicConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, c1x1,c3x3r,c3x3,c5x5r,c5x5,p):
        super(Inception, self).__init__()
        self.branch1f = BasicConv2d(in_channels, c1x1, kernel_size=1)
        self.branch2f = nn.Sequential(
            BasicConv2d(in_channels, c3x3r, kernel_size=1),
            BasicConv2d(c3x3r, c3x3, kernel_size=3,padding=1)
        )
        self.branch3f = nn.Sequential(
            BasicConv2d(in_channels, c5x5r, kernel_size=1),
            BasicConv2d(c5x5r, c5x5, kernel_size=5, padding=2)
        )
        self.branch4f = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,padding=1,stride=1),
            BasicConv2d(in_channels, p, kernel_size=1)
        )
    def forward(self,x):
        branch1 = self.branch1f(x)
        branch2 = self.branch2f(x)
        branch3 = self.branch3f(x)
        branch4 = self.branch4f(x)
        outputs = [branch1,branch2,branch3,branch4]
        return torch.cat(outputs,1)

class InceptionAux(nn.Module):
    def __init__(self, in_channel,num_class):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5,stride=3)
        self.conv = BasicConv2d(in_channel, 128,kernel_size = 1)
        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,num_class)
    def forward(self, x):
        x = self.averagePool(x)
        x = self.conv(x)
        x = torch.flatten(x,start_dim = 1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x