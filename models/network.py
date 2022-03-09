import torch
import math
import torch.nn as nn

from models.resnet import resnet50
# from models.dla import dla34


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-8
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight[None, :, None, None].expand_as(x) * x
        return out


class DnCNet(nn.Module):
    def __init__(self, base):
        super(DnCNet, self).__init__()
        self.init_layer = []

        if base == 'resnet50':
            self.base = resnet50(pretrained=True,
                                 replace_stride_with_dilation=[False, False, True])

            self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
            self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)

            self.p3_l2 = L2Norm(256, 10)
            self.p4_l2 = L2Norm(256, 10)
            self.p5_l2 = L2Norm(256, 10)

            self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
            self.feat_bn = nn.BatchNorm2d(256, momentum=0.01)
            self.feat_act = nn.ReLU(inplace=True)

            self.init_layer += [self.p3, self.p4, self.p5, self.feat]

        elif base == 'dla34':
            self.base = dla34
        else:
            raise NotImplementedError

        self.pos_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.reg_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.off_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.den_conv = nn.Conv2d(256, 1, kernel_size=1)

        self.init_layer += [self.pos_conv, self.reg_conv, self.off_conv]
        self._init_weights()

    def _init_weights(self):
        for m in self.init_layer:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.pos_conv.bias, -math.log(0.99 / 0.01))

    def forward(self, x):
        outs = self.base(x)

        p3 = self.p3(outs[-3])
        p3 = self.p3_l2(p3)
        p4 = self.p4(outs[-2])
        p4 = self.p4_l2(p4)
        p5 = self.p5(outs[-1])
        p5 = self.p5_l2(p5)
        cat = torch.cat([p3, p4, p5], dim=1)

        feat = self.feat(cat)
        feat = self.feat_bn(feat)
        feat = self.feat_act(feat)

        cls = self.pos_conv(feat)
        cls = torch.sigmoid(cls)
        reg = self.reg_conv(feat)
        off = self.off_conv(feat)
        den = self.den_conv(feat)
        den = den.abs()

        return cls, reg, off, den