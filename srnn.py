import torch
import torch.nn as nn


class srnn(nn.Module):
    def __init__(self, k, ci, c):
        super(srnn, self).__init__()
        self.k, self.ci, self.c = k, ci, c
        self.f_pad = (self.k[-1]-1)//2
        self.t_pad = self.k[0] - 1
        self.pad = nn.ConstantPad2d((self.f_pad, self.f_pad, self.t_pad, 0), value=0.)
        self.pre_conv = nn.Sequential(
            self.pad,
            nn.Conv2d(ci, c, k),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c))
        self.conv_xz = nn.Sequential(
            self.pad,
            nn.Conv2d(c, c, k))
        self.conv_xr = nn.Sequential(
            self.pad,
            nn.Conv2d(c, c, k))
        self.conv_xn = nn.Sequential(
            self.pad,
            nn.Conv2d(c, c, k))
        self.conv_hz = nn.Sequential(
            self.pad,
            nn.Conv2d(c, c, k))
        self.conv_hr = nn.Sequential(
            self.pad,
            nn.Conv2d(c, c, k))
        self.conv_hn = nn.Sequential(
            self.pad,
            nn.Conv2d(c, c, k))

    def forward(self, x, h=None):
        x = self.pre_conv(x)
        if h is None:
            z = torch.sigmoid(self.conv_xz(x))
            f = torch.tanh(self.conv_xn(x))
            h = z * f
        else:
            z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h))
            r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h))
            n = torch.tanh(self.conv_xn(x) + self.conv_hn(r * h))
            h = (1 - z) * h + z * n
        return h