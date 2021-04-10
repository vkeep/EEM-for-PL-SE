import torch
import torch.nn as nn
import numpy as np

# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

class base_net(nn.Module):
    def __init__(self, channels):
        super(base_net, self).__init__()
        self.en = Encoder(channels)
        self.de = Decoder()
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2)

    def forward(self, x):
        x, x_list = self.en(x)
        batch_num, _, seq_len, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_num, seq_len, -1)
        x, _ = self.lstm(x)
        x = x.view(batch_num, seq_len, 64, 4)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.de(x, x_list)
        del x_list
        return x

class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        en1 = nn.Sequential(
            Gate_Conv(channels, 64, kernel_size=(2, 5), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en2 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en3 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en4 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        en5 = nn.Sequential(
            Gate_Conv(64, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=0, pad=(0, 0, 1, 0)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        self.en = nn.ModuleList([en1, en2, en3, en4, en5])

    def forward(self, x):
        x_list = []
        for i in range(len(self.en)):
            x = self.en[i](x)
            x_list.append(x)
        return x, x_list


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        de1 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de2 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de3 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de4 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 3), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de5 = nn.Sequential(
            Gate_Conv(64*2, 64, kernel_size=(2, 5), stride=(1, 2), de_flag=1, chomp=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
        de6 = nn.Conv2d(64, 2, kernel_size=(1, 1))
        self.de = nn.ModuleList([de1, de2, de3, de4, de5, de6])

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            x = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de[i](x)
        x = self.de[-1](x)
        return x

class Gate_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, de_flag, pad=(0, 0, 0, 0), chomp=1):
        super(Gate_Conv, self).__init__()
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
    def forward(self, x):
        return self.conv(x) * self.gate_conv(x)

class Chomp_T(nn.Module):
    def __init__(self, t):
        super(Chomp_T, self).__init__()
        self.t = t
    def forward(self, x):
        return x[:, :, 0:-self.t, :]

# if __name__ == '__main__':
#     model = base_net()
#     model.cuda()
#     print('The number of parameters of the network is: %d' % (numParams(model)))