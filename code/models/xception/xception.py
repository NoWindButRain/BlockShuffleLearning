import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms


from .. import architectures


class Xception(architectures.Xception):
    def __init__(self, num_class=1, is_train=False):
        super(Xception, self).__init__(num_class)
        self.is_train = is_train

    def block_1(self, x: torch.Tensor):
        # 224x224
        x = self.xception.conv1(x)
        x = self.xception.bn1(x)
        x = self.xception.relu1(x)
        # 32x112x112
        x = self.xception.conv2(x)
        x = self.xception.bn2(x)
        x = self.xception.relu2(x)
        # 64x112x112
        return x

    def block_2(self, x: torch.Tensor):
        # 64x112x112
        x = self.xception.block1(x)
        # 128x56x56
        return x

    def block_3(self, x: torch.Tensor):
        # 128x56x56
        x = self.xception.block2(x)
        # 256x28x28
        return x

    def block_4(self, x: torch.Tensor):
        # 256x28x28
        x = self.xception.block3(x)
        x = self.xception.block4(x)
        x = self.xception.block5(x)
        x = self.xception.block6(x)
        x = self.xception.block7(x)
        x = self.xception.block8(x)
        x = self.xception.block9(x)
        x = self.xception.block10(x)
        x = self.xception.block11(x)
        # 728x14x14
        return x

    def block_5(self, x: torch.Tensor):
        # 728x14x14
        x = self.xception.block12(x)
        # 1024x7x7
        x = self.xception.conv3(x)
        x = self.xception.bn3(x)
        x = self.xception.relu3(x)
        # 1536x7x7
        x = self.xception.conv4(x)
        x = self.xception.bn4(x)
        x = nn.ReLU(inplace=True)(x)
        # 2048x7x7
        return x

    def forward(self, x: torch.Tensor):
        out = {}
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.xception.dropout(x)

        _out = self.xception.last_linear(x)
        out['out'] = _out
        return out


