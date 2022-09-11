import os
import time
import json
from tqdm import tqdm
import numpy as np
import torch
from sklearn import metrics

from torch.cuda.amp import autocast as autocast, GradScaler


class Train:
    def __init__(self, net, device):
        self.start_epoch = 0
        self.epoch = 0
        self.net = net
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.device = device
        self.correct_str = ''
        self.correct_str_test = ''
        self.save_path = ''
        self.writer = None
        self.best_acc = 0
        self.net_name = ''
        self.scaler = GradScaler()
        self.iter_num = 0


    def train(self, train_loader):
        self.epoch += 1
        print('\nEpoch: %d' % self.epoch)
        correct = 0
        total = 0
        train_loss = 0
        x = {}
        y = {}
        self.net.is_train = True
        self.net.train()
        self.optimizer.zero_grad()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        print(lr)
        if self.writer:
            self.writer.add_scalar('lr', lr, self.epoch)
        pbar = tqdm(train_loader(), ncols=80)
        for data in pbar:
            inputs, targets = self.pre_data(data)
            self.optimizer.zero_grad()
            with autocast():
                outputs = self.net(inputs)
                loss = self.cal_loss(outputs, targets)
                train_loss += loss.item()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            for key in outputs:
                x.setdefault(key, [])
                x[key].extend(torch.sigmoid(outputs[key]).detach().cpu().numpy())
            for key in targets:
                y.setdefault(key, [])
                y[key].extend(targets[key].cpu().numpy())
            total += inputs.size(0)
            correct += self.cal_correct(outputs, targets)
            pbar.set_description(
                self.correct_str.format(*(correct / total * 100))
            )
            self.iter_num += 1
        self.lr_scheduler.step()
        for key in outputs:
            x[key] = np.array(x[key])
        for key in targets:
            y[key] = np.array(y[key])
        self.evaluation(x, y)


    def val(self, val_loader):
        self._test(val_loader)

    def test(self, test_loader):
        self._test(test_loader, is_save=False)

    def _test(self, test_loader, is_save=True, save_txt=True):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        self.net.is_train = False
        x = {}
        y = {}
        ims = []
        with torch.no_grad():
            pbar = tqdm(test_loader(), ncols=80)
            for data in pbar:
                inputs, targets = self.pre_data(data)
                outputs = self.net(inputs)
                # loss = self.cal_loss(outputs, targets)
                # test_loss += loss.item()
                for key in outputs:
                    x.setdefault(key, [])
                    x[key].extend(torch.sigmoid(outputs[key]).cpu().numpy())
                for key in targets:
                    y.setdefault(key, [])
                    y[key].extend(targets[key].cpu().numpy())
                total += inputs.size(0)
                correct += self.cal_correct(outputs, targets)
                pbar.set_description(
                    self.correct_str_test.format(*(correct / total * 100))
                )
                self.iter_num += 1

        acc = float(correct[0] / total * 100)
        save_path = os.path.join(self.save_path, '{}_{:.6f}'.format(self.epoch, acc))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Save checkpoint.
        for key in outputs:
            x[key] = np.array(x[key])
        for key in targets:
            y[key] = np.array(y[key])
        self.evaluation(x, y)
        for key in outputs:
            x[key] = x[key].tolist()
        for key in targets:
            y[key] = y[key].tolist()
        if save_txt:
            txt_path = os.path.join(
                save_path,
                '{}.txt'.format(
                    time.strftime("%y%m%d%H%M%S", time.localtime()))
            )
            with open(txt_path, 'w') as file:
                json.dump({'x': x, 'y': y}, file)
        # if acc > best_acc and is_save:
        if is_save:
            print('Saving..')
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': self.epoch,
            }
            torch.save(
                state,
                '{}/{}_{}_{:.6f}_{}.pth'.format(
                    save_path, self.net_name, self.epoch, acc,
                    time.strftime("%y%m%d%H%M%S", time.localtime())
                )
            )
            self.best_acc = acc

    def pre_data(self, data):
        inputs = data[0]['data']
        _targets = data[0]['label']
        _targets = _targets.float().to(self.device)
        _ts = { 
            'out': _targets
        }
        return inputs, _ts

    def cal_loss(self, output, targets):
        loss = self.criterion(output['out'], targets['out'].float())
        return loss

    def cal_correct(self, outputs, targets):
        corrects = []
        out = outputs['out'].data
        out[out >= 0] = 1
        out[out < 0] = 0
        correct = (out == targets['out']).sum().item()
        corrects.append(correct)
        return np.array(corrects)

    def evaluation(self, x, y):
        x['out'] = x['out'].flatten()
        y['out'] = y['out'].flatten()
        out = x['out'].copy()
        out[out >= 0.5] = 1
        out[out <= 0.5] = 0
        print('总数: {}, 伪造: {}'.format(len(y['out']), sum(y['out'] == 1)))
        print(metrics.confusion_matrix(y['out'], out))
        acc = metrics.accuracy_score(y['out'], out)
        print('Acc: {}'.format(acc))
        auc = metrics.roc_auc_score(y['out'], x['out'])
        print('AUC: {}'.format(auc))
        print(metrics.classification_report(y['out'], out))
        return acc

    def resume(self, load_model):
        checkpoint = torch.load(load_model, map_location=self.device)
        model_dict = checkpoint['net']
        del model_dict['bs_conv7.weight']
        self.net.load_state_dict(model_dict, strict=False)
        self.best_acc = checkpoint['acc']
        self.start_epoch = checkpoint['epoch']
        self.epoch = checkpoint['epoch']
        print(self.best_acc, self.epoch)
