import time
import numpy as np
import copy

import torch
import torch.nn as nn
from networks import *
from utils.meters import *
import random

class Trainer(object):
    def __init__(self, args, model, criterion):
        super(Trainer, self).__init__()
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = criterion
        self.args = args

    def train(self, epoch, data_loaders, optimizer, print_freq=10, train_iters=400):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_meta_train = AverageMeter()
        losses_meta_test = AverageMeter()
        metaLR = optimizer.param_groups[0]['lr']

        source_count = len(data_loaders)

        end = time.time()

        for i in range(train_iters):

            # with torch.autograd.set_detect_anomaly(True):
            # divide source domains into meta_tr and meta_te
            data_loader_index = [i for i in range(source_count)]  ## 0 2
            random.shuffle(data_loader_index)
            batch_data = [data_loaders[i].next() for i in range(source_count)]
            data_time.update(time.time() - end)

            optimizer.zero_grad()
            # self_param = list(self.model.parameters())
            for p in self.model.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)

            # meta train(inner loop) begins
            loss_meta_train = 0.
            loss_meta_test = 0.
            # meta-train on samples from multiple meta train domains
            for t in range(source_count):  # 0 1
                inner_model = copy.deepcopy(self.model)
                inner_opt = torch.optim.Adam(inner_model.parameters(), lr=metaLR, weight_decay=self.args.weight_decay)

                data_time.update(time.time() - end)
                # process inputs for meta train
                traininputs = batch_data[data_loader_index[t]]
                trainid = data_loader_index[t]
                if t == len(data_loader_index)-1:
                    testinputs = batch_data[data_loader_index[0]]
                    testid = data_loader_index[0]
                else:
                    testinputs = batch_data[data_loader_index[t+1]]
                    testid = data_loader_index[t+1]
                inputs, targets = self._parse_data(traininputs)[2:]

                pred_mtr, sim_loss, sim_loss2, orth_loss = inner_model.train_forward(inputs, trainid)
                loss_mtr = self.criterion(pred_mtr, targets) + torch.sum(sim_loss) + sim_loss2 + orth_loss
         
                loss_meta_train += loss_mtr

                inner_opt.zero_grad()
                loss_mtr.backward()
                inner_opt.step()

                for p_tgt, p_src in zip(self.model.parameters(), inner_model.parameters()):
                    if p_src.grad is not None:
                        p_tgt.grad.data.add_(p_src.grad.data / source_count)

                testInputs, testMaps = self._parse_data(testinputs)[:2]
                # meta test begins
                pred_mte, sim_loss, sim_loss2, orth_loss  = inner_model.train_forward(testInputs, testid)

                loss_mte = self.criterion(pred_mte, testMaps) + torch.sum(sim_loss) + sim_loss2 + orth_loss
                loss_meta_test += loss_mte

                grad_inner_j = torch.autograd.grad(loss_mte, inner_model.parameters(), allow_unused=True)

                for p, g_j in zip(self.model.parameters(), grad_inner_j):
                    if g_j is not None:
                        p.grad.data.add_(1.0 * g_j.data / source_count)


            loss_final = loss_meta_train + loss_meta_test
            losses_meta_train.update(loss_meta_train.item())
            losses_meta_test.update(loss_meta_test.item())

            optimizer.step()

            losses.update(loss_final.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Total loss {:.3f} ({:.3f})\t'
                      'Loss {:.3f}({:.3f})\t'
                      'LossMeta {:.3f}({:.3f})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg,
                              losses_meta_train.val, losses_meta_train.avg,
                              losses_meta_test.val, losses_meta_test.avg))

    def _parse_data(self, inputs):
        imgs, dens, imgs2, dens2 = inputs
        return imgs.cuda(), dens.cuda(), imgs2.cuda(), dens2.cuda()
