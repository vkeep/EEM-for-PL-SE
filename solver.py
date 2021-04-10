import torch
import time
import os
from Backup import stage_com_mag_mse_loss
import hdf5storage
import gc
from config import stage_num, multi_gpus
tr_batch, tr_epoch,cv_epoch = [], [], []

class Solver(object):

    def __init__(self, data, model, optimizer, args):
        # load args parameters
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.loss_dir = args.loss_dir
        self.model = model
        self.optimizer = optimizer
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.best_path = args.best_path
        self.is_cp = args.is_cp
        self.cp_path = args.cp_path
        self.is_conti = args.is_conti
        self.conti_path = args.conti_path
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.print_freq = args.print_freq

        self._reset()

    def _reset(self):

        if self.is_conti:
            checkpoint = torch.load(self.conti_path)
            if multi_gpus:
                for i in range(stage_num - 1):
                    self.model[i].module.load_state_dict(checkpoint['pr{}_state_dict'.format(i + 1)])
                self.model[-1].module.load_state_dict(checkpoint['srnn_state_dict'])
            else:
                for i in range(stage_num - 1):
                    self.model[i].load_state_dict(checkpoint['pr{}_state_dict'.format(i + 1)])
                    self.model[-1].load_state_dict(checkpoint['srnn_state_dict'])

            self.start_epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.prev_cv_loss = checkpoint['cv_loss']
            self.best_cv_loss = checkpoint['best_cv_loss']
            self.having = checkpoint['having']
            self.cv_no_impv = checkpoint['cv_no_impv']
        else:
            # Reset
            self.start_epoch = 0
            self.prev_cv_loss = float("inf")
            self.best_cv_loss = float("inf")
            self.cv_no_impv = 0
            self.having = False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Begin to train.....")
            for i in range(len(self.model)):
                self.model[i].train()
            start = time.time()
            tr_avg_loss = self.run_one_epoch(epoch)
            print('-' * 90)
            print("End of Epoch %d, Time: %4f s, Train_Loss:%5f" % (int(epoch+1), time.time()-start, tr_avg_loss))
            print('-' * 90)

            # Cross cv
            print("Begin Cross Validation....")
            for i in range(len(self.model)):
                self.model[i].eval()    # BN and Dropout is off
            cv_avg_loss = self.run_one_epoch(epoch, cross_valid=True)
            print('-' * 90)
            print("Time: %4fs, CV_Loss:%5f" % (time.time() - start, cv_avg_loss))
            print('-' * 90)

            # save checkpoint
            if self.is_cp:
                check_point = {}
                check_point['epoch'] = epoch
                check_point['tr_loss'] = tr_avg_loss
                check_point['cv_loss'] = cv_avg_loss
                check_point['best_cv_loss'] = self.best_cv_loss
                check_point['cv_no_impv'] = self.cv_no_impv
                check_point['having'] = self.having
                check_point['optimizer_state_dict'] = self.optimizer.state_dict()
                if multi_gpus:
                    for i in range(stage_num-1):
                        check_point['pr{}_state_dict'.format(i+1)] = self.model[i].module.state_dict()
                    check_point['srnn_state_dict'] = self.model[-1].module.state_dict()
                else:
                    for i in range(stage_num-1):
                        check_point['pr{}_state_dict'.format(i+1)] = self.model[i].state_dict()
                    check_point['srnn_state_dict'] = self.model[-1].state_dict()
            torch.save(check_point, os.path.join(self.cp_path, 'checkpoint_epoch_%d_eem_se_model.pth.tar' % (epoch+1)))

            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = cv_avg_loss
            tr_epoch.append(tr_avg_loss)
            cv_epoch.append(cv_avg_loss)

            # save loss
            loss = {}
            loss['tr_loss'] = tr_epoch
            loss['cv_loss'] = cv_epoch
            hdf5storage.savemat(self.loss_dir, loss)

            # Adjust learning rate and early stop
            if self.half_lr:
                if cv_avg_loss >= self.prev_cv_loss:
                    self.cv_no_impv += 1
                    if self.cv_no_impv == 3:
                        self.having = True
                    if self.cv_no_impv >= 5 and self.early_stop == True:
                        print("No improvement and apply early stop")
                        break
                else:
                    self.cv_no_impv = 0

            if self.having == True:
                optim_state = self.optimizer.state_dict()
                for i in range(len(optim_state['param_groups'])):
                    optim_state['param_groups'][i]['lr'] = optim_state['param_groups'][i]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to %5f' % (optim_state['param_groups'][0]['lr']))
                self.having = False
            self.prev_cv_loss = cv_avg_loss

            if cv_avg_loss < self.best_cv_loss:
                self.best_cv_loss = cv_avg_loss
                state_list = []
                for i in range(stage_num-1):
                    state_list.append({'pr{}_state_dict': self.model[i].state_dict()})
                state_list.append({'srnn_state_dict': self.model[-1].state_dict()})
                torch.save(state_list, self.best_path)
                print("Find better cv model, saving to %s" % os.path.split(self.best_path)[-1])


    def run_one_epoch(self, epoch, cross_valid=False):
        def _batch(_, batch_info):
            batch_feat = batch_info.feats.cuda()
            batch_label = batch_info.labels.cuda()
            batch_noise = batch_info.noises.cuda()
            batch_frame_mask_list = batch_info.frame_mask_list

            # begin the forward calculation process
            esti_list = []
            h, prior_esti = None, batch_feat
            for i in range(stage_num):
                sr = torch.cat((batch_feat, prior_esti), dim=1)
                if h is None:
                    h = self.model[-1](sr)
                else:
                    h = self.model[-1](sr, h)
                prior_esti = self.model[i](h)
                esti_list.append(prior_esti)

            batch_loss = stage_com_mag_mse_loss(esti_list, batch_label, batch_noise,
                                                batch_frame_mask_list, stage_num)
            batch_loss_res = batch_loss.item()
            tr_batch.append(batch_loss_res)

            if not cross_valid:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            return batch_loss_res

        start1 = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader
        for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
            batch_loss_res = _batch(batch_id, batch_info)
            total_loss += batch_loss_res
            gc.collect()
            if batch_id % self.print_freq == 0:
                print("Epoch:%d, Iter:%d, the average_loss:%5f, current_loss:%5f, %d ms/batch."
                        % (int(epoch + 1), int(batch_id), total_loss / (batch_id + 1), batch_loss_res,
                            1000 * (time.time() - start1) / (batch_id + 1)))
        return total_loss / (batch_id + 1)