import torch.nn as nn
import torch
from Backup import numParams

class pl_lstm(nn.Module):
    def __init__(self, is_causal):
        super(pl_lstm, self).__init__()
        self.is_causal = is_causal
        self.bn1 = nn.BatchNorm1d(257)
        self.bn2 = nn.BatchNorm1d(257*2)
        self.bn3 = nn.BatchNorm1d(257*3)
        if self.is_causal is True:
            self.lstm_list = nn.ModuleList()
            self.fc_list = nn.ModuleList()
            for i in range(3):
                self.lstm_list.append(
                    nn.LSTM(257*(i+1), 1024, batch_first=True))
            for i in range(3):
                self.fc_list.append(
                    nn.Linear(1024, 257))
        else:
            self.lstm_list = nn.ModuleList()
            self.fc_list = nn.ModuleList()
            for i in range(3):
                self.lstm_list.append(
                    nn.LSTM(257 * (i + 1), 1024//2, batch_first=True, bidirectional=True))
            for i in range(3):
                self.fc_list.append(
                    nn.Linear(1024, 257))

    def forward(self, x):
        h1 = self.bn1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h1, _ = self.lstm_list[0](h1)
        h1 = self.fc_list[0](h1)
        x2 = torch.cat((x, h1), dim=-1)
        h2 = self.bn2(x2.permute(0, 2, 1)).permute(0, 2, 1)
        h2, _ = self.lstm_list[1](h2)
        h2 = self.fc_list[1](h2)
        x3 = torch.cat((x, h1, h2), dim=-1)
        h3 = self.bn3(x3.permute(0, 2, 1)).permute(0, 2, 1)
        h3, _ = self.lstm_list[-1](h3)
        h3 = self.fc_list[-1](h3)
        del x2, x3
        return [h1, h2, h3]

# if __name__ == "__main__":
#     model = pl_lstm(is_causal=True)
#     model.cuda()
#     model.train()
#     print('The number of parameters of the network is:%d' % (numParams(model)))
#     x = torch.rand(2, 100, 257).cuda()
#     x = model(x)