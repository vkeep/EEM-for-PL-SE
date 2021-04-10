import torch.nn as nn
import torch
from Backup import numParams

class pl_crn(nn.Module):
    def __init__(self, is_causal):
        super(pl_crn, self).__init__()
        self.en_list = nn.ModuleList([Encoder(1), Encoder(2), Encoder(3)])
        self.lstm_list = nn.ModuleList()
        if is_causal is True:
            for i in range(3):
                self.lstm_list.append(nn.LSTM(256, 256, num_layers=2, batch_first=True))
        else:
            for i in range(3):
                self.lstm_list.append(nn.LSTM(256, 256//2, num_layers=2, bidirectional=True, batch_first=True))
        self.de_list = nn.ModuleList([Decoder(), Decoder(), Decoder()])

    def forward(self, inpt):
        batch_num, seq_len, _ = inpt.shape
        inpt = inpt.unsqueeze(dim=1)
        x1, x1_list = self.en_list[0](inpt)
        x1 = x1.permute(0, 2, 1, 3).contiguous()
        x1 = x1.view(batch_num, seq_len, -1)
        x1, _ = self.lstm_list[0](x1)
        x1 = x1.view(batch_num, seq_len, 64, 4)
        x1 = x1.permute(0, 2, 1, 3).contiguous()
        x1 = self.de_list[0](x1, x1_list)

        x2 = torch.cat((inpt, x1), dim=1)
        x2, x2_list = self.en_list[1](x2)
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        x2 = x2.view(batch_num, seq_len, -1)
        x2, _ = self.lstm_list[1](x2)
        x2 = x2.view(batch_num, seq_len, 64, 4)
        x2 = x2.permute(0, 2, 1, 3).contiguous()
        x2 = self.de_list[1](x2, x2_list)

        x3 = torch.cat((inpt, x1, x2), dim=1)
        x3, x3_list = self.en_list[-1](x3)
        x3 = x3.permute(0, 2, 1, 3).contiguous()
        x3 = x3.view(batch_num, seq_len, -1)
        x3, _ = self.lstm_list[-1](x3)
        x3 = x3.view(batch_num, seq_len, 64, 4)
        x3 = x3.permute(0, 2, 1, 3).contiguous()
        x3 = self.de_list[-1](x3, x3_list)
        del x1_list, x2_list, x3_list
        return [x1.squeeze(dim=1), x2.squeeze(dim=1), x3.squeeze(dim=1)]

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        pad = nn.ConstantPad2d((0, 0, 1, 0), value=0.)
        en1 = nn.Sequential(
            pad,
            nn.Conv2d(in_channels, 16, kernel_size=(2, 5), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        en2 = nn.Sequential(
            pad,
            nn.Conv2d(16, 16, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        en3 = nn.Sequential(
            pad,
            nn.Conv2d(16, 16, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        en4 = nn.Sequential(
            pad,
            nn.Conv2d(16, 32, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        en5 = nn.Sequential(
            pad,
            nn.Conv2d(32, 64, kernel_size=(2, 3), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.en_list = nn.ModuleList([en1, en2, en3, en4, en5])

    def forward(self, x):
        x_list = []
        for i in range(len(self.en_list)):
            x = self.en_list[i](x)
            x_list.append(x)
        return x, x_list

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        de1 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 32, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        de2 = nn.Sequential(
            nn.ConvTranspose2d(32*2, 16, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        de3 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 16, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        de4 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 16, kernel_size=(2, 3), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        de5 = nn.Sequential(
            nn.ConvTranspose2d(16*2, 1, kernel_size=(2, 5), stride=(1, 2)),
            Chomp_T(1),
            nn.BatchNorm2d(1),
            nn.Softplus()
        )
        self.de_list = nn.ModuleList([de1, de2, de3, de4, de5])

    def forward(self, x, x_list):
        for i in range(len(x_list)):
            tmp = torch.cat((x, x_list[-(i+1)]), dim=1)
            x = self.de_list[i](tmp)
        return x


class Chomp_T(nn.Module):
    def __init__(self, chomp_t):
        super(Chomp_T, self).__init__()
        self.chomp_t = chomp_t
    def forward(self, x):
        return x[:, :, 0:-self.chomp_t, :]

# if __name__ == "__main__":
#     model = pl_crn(is_causal=True)
#     model.train()
#     model.cuda()
#     print('The number of parameters of the network is: %d' % (numParams(model)))
#     x = torch.rand(2, 100, 161).cuda()
#     x = model(x)
