import torch
from data import TrainDataset, CvDataset, TrainDataLoader, CvDataLoader
from solver import Solver
from Backup import numParams
from config import stage_num

def main(args, model):
    tr_dataset = TrainDataset(json_dir=args.json_dir,
                              batch_size=args.batch_size)
    cv_dataset = CvDataset(json_dir=args.json_dir,
                           batch_size=args.cv_batch_size)
    tr_loader = TrainDataLoader(data_set=tr_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                pin_memory=True)
    cv_loader = CvDataLoader(data_set=cv_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}

    # print base model one by one
    for i in range(len(model)):
        print(model[i])
    # count the parameter number of the network
    for i in range(len(model)):
        print('The number of trainable parameters of the %dth sub-net is:%d' % (i+1, numParams(model[i])))
    params_list, lr_list = [], []
    for i in range(stage_num):
        params_list.append(
            {'params': model[i].parameters(), 'lr':args.lr}
        )
    optimizer = torch.optim.Adam(params_list,
                                 weight_decay=args.l2)
    solver = Solver(data, model, optimizer, args)
    solver.train()

# if __name__ == '__main__':
#     args = parser.parse_args()
#     model = train_model
#     print(args)
#     main(args, model)