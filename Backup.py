"""
This script is the backup function used to support backup support for the SE system
Author: Andong Li
Time: 2019/05
"""
import torch
import torch.nn as nn
import os
import numpy as np
import sys
EPSILON = 1e-15

def stage_com_mag_mse_loss(esti_list, label, noise, frame_list, stage_num):
    mask_for_loss = []
    utt_num = label.size()[0]
    with torch.no_grad():
        for i in range(utt_num):
            tmp_mask = torch.ones((frame_list[i], label.size()[-1]), dtype=label.dtype)
            mask_for_loss.append(tmp_mask)
        mask_for_loss = nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(label.device)
        mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
    label_list = []
    cnt, loss=  0, 0.
    for i in range(stage_num):
        if i < stage_num-1:
            tmp_label = label + noise * (10 ** (-0.5*(i+1)))
        else:
            tmp_label = label
        tmp_loss = (((esti_list[i] - tmp_label) ** 2) * mask_for_loss).sum() / mask_for_loss.sum()
        cnt += i+1
        loss = loss + tmp_loss * (i+1)

    return loss / cnt + EPSILON


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num