import argparse
import torch
import torch.nn as nn
import numpy as np
from network import base_net
from srnn import srnn
from train import main
from config import *

# fix random seed
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)


parser = argparse.ArgumentParser("This project is the implementation of the manuscript 'Learning to inference with"
                                 "early-stopping for monaural speech enhancement")
parser.add_argument('--json_dir', type=str, default=json_dir,
                    help='The directory of training and validation dataset file names')
parser.add_argument('--loss_dir', type=str, default=loss_dir,
                    help='The directory to save tr loss and cv loss')
parser.add_argument('--batch_size', type=int, default=batch_size,
                    help='The number of the batch size for training dataset')
parser.add_argument('--cv_batch_size', type=int, default=batch_size,
                    help='The number of the batch size for validation dataset')
parser.add_argument('--epochs', type=int, default=epochs,
                    help='The number of the total training epoch')
parser.add_argument('--lr', type=float, default=lr,
                    help='Learning rate of the network')
parser.add_argument('--early_stop', dest='early_stop', default=1, type=int,
                    help='Early stop training when no improvement for 5 epochs')
parser.add_argument('--half_lr', type=int, default=1,
                    help='Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type=int, default=1,
                    help='Whether to shuffle within each batch')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to generate batch')
parser.add_argument('--l2', type=float, default=1e-7,
                    help='weight decay (L2 penalty)')
parser.add_argument('--best_path', default=model_best_path,
                    help='Location to save best cv model')
parser.add_argument('--is_cp', type=bool, default=is_cp)
parser.add_argument('--cp_path', type=str, default=check_point_path)
parser.add_argument('--is_conti', type=bool, default=is_conti)
parser.add_argument('--conti_path', type=str, default=conti_path,
                    help='the path to load in the checkpoint file')
parser.add_argument('--print_freq', type=int, default=100,
                    help='The frequency of printing loss infomation')


# determine the models
pr_net_list = []
for _ in range(stage_num):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if multi_gpus:
        base_model = nn.DataParallel(base_net(k, stride, c, lstm_num, is_gate, is_causal), gpu_id=gpu_id)
    base_model = base_net(k, stride, c, lstm_num, is_gate, is_causal).to(device)
    pr_net_list.append(base_model)

if multi_gpus:
    sr_model = nn.DataParallel(srnn(k, ci, c), gpu_id=gpu_id)
sr_model = srnn(k, ci, c).to(device)
pr_net_list.append(sr_model)


if __name__ == '__main__':
    args = parser.parse_args()
    model_list = pr_net_list
    print(args)
    main(args, model_list)