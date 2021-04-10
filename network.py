import torch
import torch.nn as nn
import numpy as np
from config import c, stride, k, lstm_num, is_causal

class base_net(nn.Module):
    def __init__(self, k, stride, c, lstm_num, is_gate, is_causal):
        super(base_net, self).__init__()
        self.k, self.stride, self.c = k, stride, c
        self.lstm_num = lstm_num
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.en = Encoder(k, stride, c, is_gate=is_gate)
        self.de = Decoder(k, stride, c, is_gate=is_gate)
        if is_causal:
            self.lstm = nn.LSTM(lstm_num, lstm_num, 2)
        else:
            self.lstm = nn.LSTM(lstm_num, lstm_num//2, 2, bidirectional=True)

    def forward(self, x):
        x, x_list = self.en(x)
        batch_num, int_channel, seq_len, int_feat = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_num, seq_len, -1)
        assert x.size()[-1] == self.lstm_num, 'the feat dim of x should equal to self.lstm_num'
        x, _ = self.lstm(x)
        x = x.view(batch_num, seq_len, int_channel, int_feat)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.de(x, x_list)
        del x_list
        return x

class Encoder(nn.Module):
    def __init__(self, k, stride, c, is_gate):
        super(Encoder, self).__init__()
        self.is_gate = is_gate
        self.k = k
        self.stride = stride
        self.c = c
        self.t_pad = self.k[0] - 1
        en1 = nn.Sequential(
            Gate_Conv(c, c, (2, 5), stride, de_flag=0, pad=(0,0,self.t_pad,0), gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        en2 = nn.Sequential(
            Gate_Conv(c, c, k, stride, de_flag=0, pad=(0,0,self.t_pad,0), gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        en3 = nn.Sequential(
            Gate_Conv(c, c, k, stride, de_flag=0, pad=(0,0,self.t_pad,0), gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        en4 = nn.Sequential(
            Gate_Conv(c, c, k, stride, de_flag=0, pad=(0,0,self.t_pad,0), gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        en5 = nn.Sequential(
            Gate_Conv(c, c, k, stride, de_flag=0, pad=(0,0,self.t_pad,0), gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        self.en = nn.ModuleList([en1, en2, en3, en4, en5])

    def forward(self, x):
        x_list = []
        for i in range(len(self.en)):
            x = self.en[i](x)
            x_list.append(x)
        return x, x_list


class Decoder(nn.Module):
    def __init__(self, k, stride, c, is_gate):
        super(Decoder, self).__init__()
        self.k, self.stride, self.c, self.is_gate = k, stride, c, is_gate
        self.t_chomp = self.k[0] - 1
        de1 = nn.Sequential(
            Gate_Conv(c*2, c, k, stride, de_flag=1, chomp=self.t_chomp, gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        de2 = nn.Sequential(
            Gate_Conv(c*2, c, k, stride, de_flag=1, chomp=self.t_chomp, gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        de3 = nn.Sequential(
            Gate_Conv(c*2, c, k, stride, de_flag=1, chomp=self.t_chomp, gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        de4 = nn.Sequential(
            Gate_Conv(c*2, c, k, stride, de_flag=1, chomp=self.t_chomp, gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        de5 = nn.Sequential(
            Gate_Conv(c*2, c, (2, 5), stride, de_flag=1, chomp=self.t_chomp, gate=is_gate),
            nn.InstanceNorm2d(c, True),
            nn.PReLU(c)
        )
        de6 = nn.Conv2d(c, 2, (1, 1))
        self.de = nn.ModuleList([de1, de2, de3, de4, de5, de6])

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de[i](x)
        x = self.de[-1](x)
        return x


class Gate_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag, pad=(0, 0, 0, 0), chomp=1, gate=True):
        super(Gate_Conv, self).__init__()
        self.gate = gate
        if gate:
            if de_flag == 0:
                self.conv = nn.Sequential(
                    nn.ConstantPad2d(pad, value=0.),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride))
                self.gate_conv = nn.Sequential(
                    nn.ConstantPad2d(pad, value=0.),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride),
                    nn.Sigmoid())
            else:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride),
                    Chomp_T(chomp))
                self.gate_conv = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels= out_channels,
                                    kernel_size=kernel_size, stride=stride),
                    Chomp_T(chomp),
                    nn.Sigmoid())
        else:
            if de_flag == 0:
                self.conv = nn.Sequential(
                    nn.ConstantPad2d(pad, value=0.),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride))
            else:
                self.conv = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride),
                    Chomp_T(chomp))

    def forward(self, x):
        if self.gate:
            x = self.conv(x) * self.gate_conv(x)
        else:
            x = self.conv(x)
        return x

class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t
    def forward(self, x):
        return x[:, :, 0:-self.t, :]

# if __name__ == '__main__':
#     model = base_net((2, 3), (1, 2), 64, 256, False)
#     model.eval()
#     model.cuda()
#     x = torch.FloatTensor(4, 64, 10, 161).cuda()
#     x = model(x)