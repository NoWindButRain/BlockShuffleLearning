import torch
from torch import nn as nn
from torch.nn import functional as F
from .renet import ResNet50


class ResNet50BSL(ResNet50):
    def __init__(
            self, num_class=1, is_train=False,
            pretrained=True, progress=True,
            is_bs_adv=False, is_rs_adv=False, **kwargs):
        super(ResNet50BSL, self).__init__(num_class, is_train, pretrained, progress)
        self.is_train = is_train
        self.is_bs_adv = is_bs_adv
        self.is_rs_adv = is_rs_adv
        self.bs_dropout7 = nn.Dropout2d(p=0.5)
        self.bs_conv7 = nn.Conv2d(2048, 1, (1, 1), bias=False)
        self.rs_linear = nn.Linear(2048, 1)
        self.rs_dropout7 = nn.Dropout2d(p=0.1)
        self.rs_conv7 = nn.Conv2d(2048, 2, (1, 1), bias=False)
        self.rs_ht = torch.nn.Hardtanh(min_val=-1, max_val=1)

    def forward(self, x):
        out = {}
        x = self.resnet.conv1(x)
        # 64, 112, 112
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        # 64, 56, 56

        x = self.resnet.layer1(x)
        # 256, 56, 56
        x = self.resnet.layer2(x)
        # 512, 28, 28
        x = self.resnet.layer3(x)
        # 1024, 14, 14
        x = self.resnet.layer4(x)
        # 2048, 7, 7
        if self.is_train:
            if self.is_bs_adv:
                bs2 = self.bs_conv7(self.bs_dropout7(x))
                bs2 = torch.flatten(bs2, 1)
                out['bs2'] = bs2
            if self.is_rs_adv:
                rsi = self.rs_conv7(self.rs_dropout7(x))
                rsi = self.rs_ht(rsi)
                rsi = torch.flatten(rsi, 1)
                out['rsi'] = rsi

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        if self.is_train and self.is_rs_adv:
            rs = self.rs_linear(x)
            out['rs'] = rs
        x = self.resnet.last_linear(x)
        out['out'] = x
        return out
















