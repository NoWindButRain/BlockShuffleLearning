import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models import resnet

from .. import architectures


class ResNet50(architectures.fornet.FeatureExtractor):
    def __init__(self, num_class=1, is_train=False, pretrained=True, progress=True, **kwargs):
        super(ResNet50, self).__init__()
        self.is_train = is_train
        self.resnet = resnet.ResNet(
            resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
        if pretrained:
            state_dict = resnet.load_state_dict_from_url(
                resnet.model_urls['resnet50'],
                progress=progress)
            self.resnet.load_state_dict(state_dict)
        self.resnet.last_linear = nn.Linear(
            512 * resnet.Bottleneck.expansion, num_class)
        self.dropout = nn.Dropout()

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

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.resnet.last_linear(x)
        out['out'] = x
        return out
















