import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms

from tools.train import Train
from models.xception import Xception


device_id = 0
device = torch.device('cuda:{}'.format(device_id)) if torch.cuda.is_available() else torch.device('cpu')
is_train = True
is_resume = True
load_model = \
'/root/workspace/bsl/checkpoints/xception_ff_c23_classic_split/8_96.659195/xception_8_96.659195_220329014946.pth'

print('Init model...')
net = Xception(num_class=1, is_train=is_train).eval().to(device)

train = Train(net, device)
train.net_name = 'xception'
train.correct_str_test = '{:.2f}'

if is_train:
    train.correct_str = '{:.2f}'
    save_path = '../checkpoints/xception_299_ff_c23_classic_split/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, os.path.split(__file__)[-1])
    shutil.copy(__file__, file_path)
    train.save_path = save_path
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(net.get_trainable_parameters(), lr=1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        cooldown=2 * 10,
        min_lr=1e-5 * 1e-5,
    )
    train.criterion = criterion
    train.optimizer = optimizer
    train.lr_scheduler = lr_scheduler

if is_resume:
    print('Load weight...')
    train.resume(load_model)

print('Init data...')
from configs import ff_test

def train_loader():
    return ff_test.dataloader


def val_loader():
    return ff_test.dataloader


def test_loader():
    return ff_test.dataloader




