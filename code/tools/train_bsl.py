import datetime
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from .train import Train


class TrainBSL(Train):
    def __init__(self, net, device):
        super(TrainBSL, self).__init__(net, device)
        self.region_num = 7
        self.shuffle_size = 16

    def pre_data(self, data):
        inputs = data[0].to(self.device)
        _targets = data[1]
        _targets = (_targets).float().to(self.device)
        _ts = {
            'out': _targets,
        }
        if self.net.is_train:
            images = []
            rs = []
            rsi = []
            bs = []
            for image in inputs:
                image, is_region_shuffle, bsindex, rsindex = self._bs_image(image)
                images.append(image)
                rs.append(is_region_shuffle)
                bs.append(bsindex)
                rsi.append(rsindex)
            inputs = torch.stack(images, dim=0).to(self.device)
            rs = torch.stack(rs, dim=0).to(self.device)
            rsi = torch.stack(rsi, dim=0).to(self.device)
            bs = torch.stack(bs, dim=0).to(self.device)
            _ts['rs'] = rs
            _ts['rsi'] = rsi
            _ts['bs'] = bs
        if self.iter_num % 20 == 0:
            _timg = (inputs[0] * 0.5 * 255 + 127).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            fig = plt.gcf()
            fig.clf()
            fig.set_size_inches(5, 5)
            plt.imshow(_timg)
            if self.writer:
                self.writer.add_figure("figure/input", plt.gcf(), self.iter_num)
        return inputs, _ts

    def cal_loss(self, output, targets):
        loss = self.criterion(output['out'], targets['out'].float())
        if self.writer:
            self.writer.add_scalar('loss/cls', loss, self.iter_num)
        if self.net.is_train:
            if self.net.is_rs_adv:
                loss_rs = self.criterion(output['rs'], targets['rs'].float())
                loss_rsi = self.criterion_rs(output['rsi'], targets['rsi'])
                if self.writer:
                    self.writer.add_scalar('loss/rsi', loss_rsi, self.iter_num)
                loss += 1 * (loss_rs + loss_rsi)
            if self.net.is_bs_adv:
                loss_bs2 = self.criterion_bs(output['bs2'], targets['bs'])
                if self.writer:
                    self.writer.add_scalar('loss/bs', loss_bs2, self.iter_num)
                loss += 0.01 * loss_bs2
        return loss

    def cal_correct(self, outputs, targets):
        corrects = []
        out = outputs['out'].data
        out[out >= 0] = 1
        out[out < 0] = 0
        correct = (out == targets['out']).sum().item()
        corrects.append(correct)
        if not self.net.is_train:
            return np.array(corrects)
        if self.net.is_rs_adv:
            rs = outputs['rs'].data
            rs[rs >= 0] = 1
            rs[rs < 0] = 0
            correct = (rs == targets['rs']).sum().item()
            corrects.append(correct)
        if self.net.is_bs_adv:
            bs2 = outputs['bs2'].data
            bs2[bs2 >= 0] = 1
            bs2[bs2 < 0] = 0
            if self.iter_num % 20 == 0:
                fig = plt.gcf()
                fig.clf()
                fig.set_size_inches(10, 4)
                plt.subplot(1, 2, 1)
                _bs = bs2[0].detach().cpu().view((14, 14)).numpy()
                sns.heatmap(_bs, annot=True)
                plt.subplot(1, 2, 2)
                _bs = targets['bs'][0].detach().cpu().view((14, 14)).numpy()
                sns.heatmap(_bs, annot=True)
                if self.writer:
                    self.writer.add_figure("figure/bs", plt.gcf(), self.iter_num)
            correct = (bs2 == targets['bs']).float().mean(dim=(-1)).sum().item()
            corrects.append(correct)
        if self.net.is_rs_adv:
            rsi = outputs['rsi'].data
            if self.iter_num % 20 == 0:
                fig = plt.gcf()
                fig.clf()
                fig.set_size_inches(8, 8)
                _rs1 = rsi[0].detach().cpu()[: 49].view((7, 7)).numpy()
                _rs1 = (_rs1 + 1) / 2 * 6
                _rs2 = rsi[0].detach().cpu()[49: ].view((7, 7)).numpy()
                _rs2 = (_rs2 + 1) / 2 * 6
                plt.subplot(2, 2, 1)
                sns.heatmap(_rs1, annot=True)
                plt.subplot(2, 2, 2)
                sns.heatmap(_rs2, annot=True)
                _bs = targets['bs'][0].detach().cpu().view((14, 14)).numpy()
                _rs1 = targets['rsi'][0].detach().cpu()[: 49].view((7, 7)).numpy()
                _rs1 = (_rs1 + 1) / 2 * 6
                _rs2 = targets['rsi'][0].detach().cpu()[49:].view((7, 7)).numpy()
                _rs2 = (_rs2 + 1) / 2 * 6
                plt.subplot(2, 2, 3)
                sns.heatmap(_rs1, annot=True)
                plt.subplot(2, 2, 4)
                sns.heatmap(_rs2, annot=True)
                if self.writer:
                    self.writer.add_figure("figure/rs", plt.gcf(), self.iter_num)
            _c = (rsi - targets['rsi']).abs()
            correct = _c.mean(dim=-1).sum().item()
            corrects.append(correct)
        return np.array(corrects)

    def _bs_image(self, image):
        region_num = self.region_num
        shuffle_size = self.shuffle_size
        # _shuffle = min(4, 1 + self.epoch)
        _shuffle = 4 - self.epoch % 3
        is_region_shuffle = np.random.randint(1000) % _shuffle
        is_block_shuffle = np.random.randint(1000) % _shuffle
        # is_region_shuffle = 1
        # is_block_shuffle = 0
        if region_num != 0 and is_region_shuffle == 0:
            image, rsindex = self.region_shuffle(image, region_num=region_num)
            rsindex = torch.FloatTensor(rsindex)
        else:
            _index = np.arange(region_num * region_num)
            rsindex = np.concatenate([_index // region_num, _index % region_num]) / (region_num - 1) * 2 - 1
            rsindex = torch.FloatTensor(rsindex)
        is_region_shuffle = torch.LongTensor([int(is_region_shuffle == 0 and region_num != 0), ])
        bsindex = 0
        if shuffle_size != 0 and is_block_shuffle == 0:
            _p = 0.4 / (is_region_shuffle.item() + 1)
            image, bsindex = self.block_shuffle(image, size=shuffle_size, ratio=_p)
            bsindex = torch.FloatTensor(bsindex)
        else:
            _index = np.zeros(224//shuffle_size*224//shuffle_size)
            bsindex = torch.FloatTensor(_index)
        return image, is_region_shuffle, bsindex, rsindex

    def block_shuffle(self, img, size=2, ratio=0.4):
        bnum = 224 // size
        bsindex = np.random.randint(1000, size=(bnum * bnum)) <= 1000 * (ratio * np.random.rand() + 0.1)
        index = [np.arange(size * size) for i in range(sum(bsindex))]
        for i in index:
            np.random.shuffle(i)
        _img = img.reshape((3, 224, 224 // size, size)).transpose(1, 2) \
            .reshape((3, 224 // size, 224 // size, size * size))
        j = 0
        for i in range(len(bsindex)):
            if not bsindex[i]:
                continue
            _img[:, i // bnum, i % bnum] = _img[:, i // bnum, i % bnum, index[j]]
            j += 1
        return _img.reshape((3, 224 // size, 224, size)).transpose(1, 2).reshape((3, 224, 224)), bsindex

    def region_shuffle(self, img, region_num=7):
        index = np.arange(region_num * region_num)
        np.random.shuffle(index)
        rs_image = img.reshape((3, 224, region_num, 224 // region_num)).transpose(1, 2) \
            .reshape((3, region_num, region_num, (224 // region_num) * (224 // region_num))) \
            .transpose(2, 3).transpose(1, 2) \
            .reshape((3, (224 // region_num) * (224 // region_num), region_num * region_num))[:, :, index] \
            .reshape((3, (224 // region_num) * (224 // region_num), region_num, region_num))\
            .transpose(1, 2).transpose(2, 3) \
            .reshape((3, region_num, 224, 224 // region_num)).transpose(1, 2) \
            .reshape((3, 224, 224))
        rsindex = np.concatenate([index // region_num, index % region_num]) / (region_num - 1) * 2 - 1
        return rs_image, rsindex