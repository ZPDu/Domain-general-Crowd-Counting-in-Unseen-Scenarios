from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
import numpy as np
from utils.meters import AverageMeter

def evaluate_all(model, data_loader, print_freq=50):
    model.eval()
    mseloss = torch.nn.MSELoss()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    MAEs = 0.
    MSEs = 0.
    val_loss = []

    end = time.time()
    with torch.no_grad():
        for i, (imgs, gts) in enumerate(data_loader):
            data_time.update(time.time() - end)
            imgs = imgs.cuda()
            gts = gts.cuda()
            dens = model(imgs)
            for j, (den, gt) in enumerate(zip(dens, gts)):
                loss = mseloss(den, gt)
                val_loss.append(loss.item())

                den = torch.sum(den)/1000.
                gt = torch.sum(gt)/1000.
                MAEs += abs(gt - den)
                MSEs += ((gt-den)*(gt-den))

            batch_time.update(time.time() - end)
            end = time.time()

            # if (i + 1) % print_freq == 0:
            #     print('Predict dens: [{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           .format(i + 1, len(data_loader),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg))
    mae = MAEs / len(data_loader)
    mse = torch.sqrt(MSEs / len(data_loader))
    loss = torch.mean(torch.Tensor(val_loss))
    print('mae:', mae, 'mse:', mse, 'loss:', loss)

    return mae, mse

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader):

        return evaluate_all(self.model, data_loader)