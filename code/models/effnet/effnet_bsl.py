import torch
from torch import nn as nn
from torch.nn import functional as F
from .effnet import Effb4Net


class Effb4NetBSL(Effb4Net):
    def __init__(
            self, num_class=1, is_train=False,
            pretrained=True, progress=True,
            is_bs_adv=False, is_rs_adv=False, **kwargs):
        super(Effb4NetBSL, self).__init__(num_class, is_train, pretrained, progress)
        self.is_train = is_train
        self.is_bs_adv = is_bs_adv
        self.is_rs_adv = is_rs_adv
        self.bs_dropout7 = nn.Dropout2d(p=0.5)
        self.bs_conv7 = nn.Conv2d(1792, 1, (1, 1), bias=False)
        self.rs_linear = nn.Linear(1792, 1)
        self.rs_dropout7 = nn.Dropout2d(p=0.1)
        self.rs_conv7 = nn.Conv2d(1792, 2, (1, 1), bias=False)
        self.rs_ht = torch.nn.Hardtanh(min_val=-1, max_val=1)

    def forward(self, x):
        out = {}
        x = self.effnet.conv_first(x)
        # 48, 112, 112
        for i in range(len(self.effnet.inverted_residual_setting)):
            x = getattr(self.effnet, 'layer%d' % (i + 1))(x)
            # 24, 112, 112
            # 32, 56, 56
            # 56, 28, 28
            # 112, 14, 14
            # 160, 14, 14
            # 272, 7, 7
            # 448, 7, 7
        x = self.effnet.conv_last(x)
        # 1792, 7, 7
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

        x = torch.mean(x, dim=(2, 3))
        # 1792
        x = self.effnet.dropout(x)
        if self.is_train and self.is_rs_adv:
            rs = self.rs_linear(x)
            out['rs'] = rs
        x = self.effnet.fc(x)
        out['out'] = x
        return out


