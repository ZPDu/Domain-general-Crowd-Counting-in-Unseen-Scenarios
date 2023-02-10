from torchvision import models
from torch import nn
import torch
import torch.functional as F
from .MetaModule import *
import math

def upsample_bilinear(x, size):
    return F.interpolate(x, size=size[2:], mode='bilinear', align_corners=False)

class Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super(Backbone, self).__init__()

        # frontend
        model = list(models.vgg16(pretrained=pretrained).features.children())
        self.feblock1 = nn.Sequential(*model[:16])
        self.feblock2 = nn.Sequential(*model[16:23])
        self.feblock3 = nn.Sequential(*model[23:30])

        # backend
        self.beblock3 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock2 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.beblock1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.feblock1(x)
        x1 = x
        x = self.feblock2(x)
        x2 = x
        x = self.feblock3(x)

        # decoding stage
        x = self.beblock3(x)
        x3_ = x
        x = upsample_bilinear(x, x2.shape)
        x = torch.cat([x, x2], 1)

        x = self.beblock2(x)
        x2_ = x
        x = upsample_bilinear(x, x1.shape)
        x = torch.cat([x, x1], 1)

        x1_ = self.beblock1(x)

        x2_ = upsample_bilinear(x2_, x1.shape)
        x3_ = upsample_bilinear(x3_, x1.shape)

        x = torch.cat([x1_, x2_, x3_], 1)
        return x


class MetaMSNetBase(MetaModule):
    def __init__(self, pretrained=False):
        super(MetaMSNetBase, self).__init__()

        self.backbone = Backbone(True)

        self.output_layer = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
            # nn.Dropout2d(p=0.5)
        )

        # self._initialize_weights()
        self.part_num = 1024
        variance = math.sqrt(1.0)
        self.sem_mem = nn.Parameter(torch.FloatTensor(1, 256, self.part_num).normal_(0.0, variance))
        self.sty_mem = nn.Parameter(torch.FloatTensor(4, 1, 256, self.part_num // 4).normal_(0.0, variance))
        self.sem_down = nn.Sequential(
            nn.Conv2d(512 + 256 + 128, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.sty_down = nn.Sequential(
            nn.Conv2d(512 + 256 + 128, 256, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_features(self, x):
        x = self.backbone(x)
        feature = self.sty_down(x)

        return feature.unsqueeze(0)

    # same as network_forward
    def train_forward(self, x, label):
        size = x.shape
        # encoding stage

        x = self.backbone(x)
        # memory:
        memory = self.sem_mem.repeat(x.shape[0], 1, 1)
        memory_key = memory.transpose(1, 2)  # bs*50*256
        sem_pre = self.sem_down(x)
        sty_pre = self.sty_down(x)

        sem_pre_ = sem_pre.view(sem_pre.shape[0], sem_pre.shape[1], -1)
        diLogits = torch.bmm(memory_key, sem_pre_)  # bs*50*(h*w)
        invariant_feature = torch.bmm(memory_key.transpose(1, 2), F.softmax(diLogits, dim=1))

        # calculate rec loss
        recon_sim = torch.bmm(invariant_feature.transpose(1, 2), sem_pre_)
        sim_gt = torch.linspace(0, sem_pre.shape[2] * sem_pre.shape[3] - 1,
                                sem_pre.shape[2] * sem_pre.shape[3]).unsqueeze(0).repeat(sem_pre.shape[0], 1).cuda()
        sim_loss = F.cross_entropy(recon_sim, sim_gt.long(), reduction='none') * 0.1

        invariant_feature_ = invariant_feature.view(invariant_feature.shape[0], invariant_feature.shape[1],
                                                   sem_pre.shape[2], sem_pre.shape[3])

        # density prediction
        den = self.output_layer(invariant_feature_)
        den = upsample_bilinear(den, size=size)

        # re-encoding ds features
        memory2 = self.sty_mem[label].cuda()
        memory2 = memory2.repeat(x.shape[0], 1, 1)
        mem2_key = memory2.transpose(1, 2)
        sty_pre_ = sty_pre.view(sty_pre.shape[0], sty_pre.shape[1], -1)
        dsLogits = torch.bmm(mem2_key, sty_pre_)
        spe_feature = torch.bmm(mem2_key.transpose(1, 2), F.softmax(dsLogits, dim=1))
        # calculate rec loss with style features
        recon_sim2 = torch.bmm(spe_feature.transpose(1, 2), sty_pre_)
        sim_gt2 = torch.linspace(0, sty_pre.shape[2] * sty_pre.shape[3] - 1,
                                 sty_pre.shape[2] * sty_pre.shape[3]).unsqueeze(0).repeat(sty_pre.shape[0], 1).cuda()
        sim_loss2 = F.cross_entropy(recon_sim2, sim_gt2.long(), reduction='sum') * 0.1
        # orthogonal loss between sem and sty features
        orth_pre = torch.bmm(sty_pre_.transpose(1, 2), sem_pre_)
        # orth = torch.bmm(spe_feature.transpose(1, 2), invariant_feature)
        orth_loss = 0.01 * torch.sum(torch.pow(torch.diagonal(orth_pre, dim1=-2, dim2=-1), 2))

        return den, sim_loss, sim_loss2, orth_loss

    def forward(self, x):
        size = x.shape

        x = self.backbone(x)

        # memory:
        memory = self.sem_mem.repeat(x.shape[0], 1, 1)
        memory_key = memory.transpose(1, 2)  # bs*50*256
        sem_pre = self.sem_down(x)

        sem_pre_ = sem_pre.view(sem_pre.shape[0], sem_pre.shape[1], -1)
        diLogits = torch.bmm(memory_key, sem_pre_)  # bs*50*(h*w)
        invariant_feature = torch.bmm(memory_key.transpose(1, 2), F.softmax(diLogits, dim=1))

        invariant_feature = invariant_feature.view(invariant_feature.shape[0], invariant_feature.shape[1],
                                                   sem_pre.shape[2], sem_pre.shape[3])

        den = self.output_layer(invariant_feature)

        den = upsample_bilinear(den, size=size)

        return den

class MetaMemNet(MetaModule):
    def getBase(self):
        baseModel = MetaMSNetBase(True)
        return baseModel

    def __init__(self):
        super(MetaMemNet, self).__init__()
        self.base = self.getBase()

    def train_forward(self, x, label):
        dens, sim_loss, sim_loss2, orth_loss = self.base.train_forward(x, label)

        dens = upsample_bilinear(dens, x.shape)

        return dens, sim_loss, sim_loss2, orth_loss

    def forward(self, x):
        dens = self.base(x)

        dens = upsample_bilinear(dens, x.shape)

        return dens

    def get_grads(self):
        grads = []
        for p in self.base.parameters():
            grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for k, p in enumerate(self.base.parameters()):
            dims = p.shape
            end = start + dims.numel()
            p.grad.data = new_grads[start:end].reshape(dims)
            start = end

    def conv_features(self, x):
        x = self.base.conv_features(x)

        return x
