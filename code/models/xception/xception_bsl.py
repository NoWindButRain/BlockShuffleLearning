import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms

from .xception import Xception


class XceptionBSL(Xception):
    def __init__(
            self, num_class=1, is_train=False, is_bs_adv=False, is_rs_adv=False,
            shuffle_size=16, region_num=7):
        super(XceptionBSL, self).__init__(num_class)
        self.is_train = is_train
        self.is_bs_adv = is_bs_adv
        self.is_rs_adv = is_rs_adv
        self.shuffle_size = shuffle_size
        self.region_num = region_num
        _r = 32 // self.shuffle_size
        self.pixleshuffle = nn.PixelShuffle(_r)
        self.bs_conv7 = nn.Conv2d(2048 // (_r*_r), 1, (1, 1), bias=False)
        self.rs_linear = nn.Linear(2048, 1)
        _r = self.region_num // 7
        self.pixleshuffle2 = nn.PixelShuffle(_r)
        self.rs_conv7 = nn.Conv2d(2048 // (_r*_r), 2, (1, 1), bias=False)
        self.rs_ht = torch.nn.Hardtanh(min_val=-1, max_val=1)

    def forward(self, x: torch.Tensor):
        out = {}
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        if self.is_train:
            if self.is_bs_adv:
                bs2 = self.pixleshuffle(x)
                bs2 = self.bs_conv7(bs2)
                bs2 = torch.flatten(bs2, 1)
                out['bs2'] = bs2
            if self.is_rs_adv:
                rsi = self.pixleshuffle2(x)
                rsi = self.rs_conv7(rsi)
                rsi = self.rs_ht(rsi)
                rsi = torch.flatten(rsi, 1)
                out['rsi'] = rsi
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.xception.dropout(x)
        if self.is_train and self.is_rs_adv:
            rs = self.rs_linear(x)
            out['rs'] = rs
        _out = self.xception.last_linear(x)
        out['out'] = _out
        return out



