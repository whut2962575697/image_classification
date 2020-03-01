# -*- encoding: utf-8 -*-
'''
@File    :   trainer.py
@Contact :   whut.hexin@foxmail.com
@License :   (C)Copyright 2017-2018, HeXin

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/1/28 13:20   xin      1.0         None
'''

from utils import AvgerageMeter, make_optimizer, calculate_score, mixup_data, rand_bbox
from common.sync_bn import convert_model
from common.warmup import WarmupMultiStepLR, GradualWarmupScheduler



import logging
import os
from tensorboardX import SummaryWriter
from torch import nn
import torch
import os.path as osp
from tqdm import tqdm
import numpy as np


class BaseTrainer(object):
    def __init__(self, cfg, model, train_dl, val_dl,
                                  loss_func, num_gpus, device):

        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss_func = loss_func

        self.loss_avg = AvgerageMeter()
        self.acc_avg = AvgerageMeter()
        self.f1_avg = AvgerageMeter()

        self.val_loss_avg = AvgerageMeter()
        self.val_acc_avg = AvgerageMeter()
        self.device = device

        self.train_epoch = 1

        if cfg.SOLVER.USE_WARMUP:
            self.optim = make_optimizer(self.model, opt=self.cfg.SOLVER.OPTIMIZER_NAME, lr=cfg.SOLVER.BASE_LR * 0.1,
                                        weight_decay=self.cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)
        else:
            self.optim = make_optimizer(self.model, opt=self.cfg.SOLVER.OPTIMIZER_NAME, lr=cfg.SOLVER.BASE_LR,
                                        weight_decay=self.cfg.SOLVER.WEIGHT_DECAY, momentum=0.9)
        if cfg.SOLVER.RESUME:
            print("Resume from checkpoint...")
            checkpoint = torch.load(cfg.SOLVER.RESUME_CHECKPOINT)
            param_dict = checkpoint['model_state_dict']
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optim.state.values():
                for k, v in state.items():
                    print(type(v))
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            self.train_epoch = checkpoint['epoch']+1
            for i in param_dict:
                if i.startswith("module"):
                    new_i = i[7:]
                else:
                    new_i = i
                if 'classifier' in i or 'fc' in i:
                    continue
                self.model.state_dict()[new_i].copy_(param_dict[i])

        self.batch_cnt = 0

        self.logger = logging.getLogger('baseline.train')
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR

        self.epochs = cfg.SOLVER.MAX_EPOCHS

        if cfg.SOLVER.TENSORBOARD.USE:
            summary_dir = os.path.join(cfg.OUTPUT_DIR, 'summaries/')
            os.makedirs(summary_dir, exist_ok=True)
            self.summary_writer = SummaryWriter(log_dir=summary_dir)
        self.current_iteration = 0

        self.logger.info(self.model)

        if self.cfg.SOLVER.USE_WARMUP:

            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.epochs, eta_min=cfg.SOLVER.MIN_LR)
            self.scheduler = GradualWarmupScheduler(self.optim, multiplier=10, total_epoch=cfg.SOLVER.WARMUP_EPOCH,
                                                      after_scheduler=scheduler_cosine)
            # self.scheduler = WarmupMultiStepLR(self.optim, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
            #                                cfg.SOLVER.WARMUP_EPOCH, cfg.SOLVER.WARMUP_METHOD)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, self.epochs, eta_min=cfg.SOLVER.MIN_LR)

        if num_gpus > 1:

            self.logger.info(self.optim)
            self.model = nn.DataParallel(self.model)
            if cfg.SOLVER.SYNCBN:
                self.model = convert_model(self.model)
                self.model = self.model.to(device)
                self.logger.info('More than one gpu used, convert model to use SyncBN.')
                self.logger.info('Using pytorch SyncBN implementation')
                self.logger.info(self.model)

            self.logger.info('Trainer Built')

            return

        else:
            self.model = self.model.to(device)
            self.logger.info('Cpu used.')
            self.logger.info(self.model)
            self.logger.info('Trainer Built')

            return

    def handle_new_batch(self):

        lr = self.scheduler.get_lr()[0]
        if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
            if self.summary_writer:
                self.summary_writer.add_scalar('Train/lr', lr, self.current_iteration)
                self.summary_writer.add_scalar('Train/loss', self.loss_avg.avg, self.current_iteration)
                self.summary_writer.add_scalar('Train/acc', self.acc_avg.avg, self.current_iteration)
                self.summary_writer.add_scalar('Train/f1', self.f1_avg.avg, self.current_iteration)


        self.batch_cnt += 1
        self.current_iteration += 1
        if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:

            self.logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f},'
                             'acc: {:.3f}, f1: {:.3f}, Base Lr: {:.2e}'
                             .format(self.train_epoch, self.batch_cnt,
                                     len(self.train_dl), self.loss_avg.avg,
                                     self.acc_avg.avg, self.f1_avg.avg, lr))

    def handle_new_epoch(self):


        self.batch_cnt = 1

        self.logger.info('Epoch {} done'.format(self.train_epoch))
        self.logger.info('-' * 20)
        checkpoint = {
            'epoch': self.train_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }
        torch.save(checkpoint, osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch_last.pth'))
        # torch.save(self.optim.state_dict(), osp.join(self.output_dir,
        #                                              self.cfg.MODEL.NAME +"_k_f"+str(self.cfg.DATALOADER.VAL_FOLDER)+  '_epoch_last_optim.pth'))

        if self.train_epoch > self.cfg.SOLVER.START_SAVE_EPOCH and self.train_epoch % self.checkpoint_period == 0:
            self.save()
        if (self.train_epoch > 0 and self.train_epoch % self.eval_period == 0) or self.train_epoch == 1:
            self.evaluate()

        self.acc_avg.reset()
        self.f1_avg.reset()
        self.loss_avg.reset()
        self.val_loss_avg.reset()


        self.scheduler.step()
        self.train_epoch += 1

    def step(self, batch):
        self.model.train()
        self.optim.zero_grad()
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)

        if self.cfg.INPUT.USE_MIX_UP:
            data, target_a, target_b, lam = mixup_data(data, target, 0.4, True)
        self.use_cut_mix = False
        if self.cfg.INPUT.USE_RICAP:
            I_x, I_y = input.size()[2:]

            w = int(np.round(I_x * np.random.beta(args.ricap_beta, args.ricap_beta)))
            h = int(np.round(I_y * np.random.beta(args.ricap_beta, args.ricap_beta)))
            w_ = [w, I_x - w, w, I_x - w]
            h_ = [h, h, I_y - h, I_y - h]

            cropped_images = {}
            c_ = {}
            W_ = {}
            for k in range(4):
                idx = torch.randperm(input.size(0))
                x_k = np.random.randint(0, I_x - w_[k] + 1)
                y_k = np.random.randint(0, I_y - h_[k] + 1)
                cropped_images[k] = input[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
                c_[k] = target[idx].cuda()
                W_[k] = w_[k] * h_[k] / (I_x * I_y)

            patched_images = torch.cat(
                (torch.cat((cropped_images[0], cropped_images[1]), 2),
                 torch.cat((cropped_images[2], cropped_images[3]), 2)),
                3)
            data = patched_images.to(self.device)




        if self.cfg.INPUT.USE_CUT_MIX:
            r = np.random.rand(1)
            if r < 0.5:
                self.use_cut_mix = True
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(data.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
                # compute output
                data = torch.autograd.Variable(data, requires_grad=True)
                target_a_var = torch.autograd.Variable(target_a)
                target_b_var = torch.autograd.Variable(target_b)

        outputs = self.model(data)


        # loss = self.loss_func(outputs, target)
        if self.cfg.INPUT.USE_RICAP:
            loss = sum([W_[k] * self.loss_func(outputs, c_[k]) for k in range(4)])
        elif self.cfg.INPUT.USE_MIX_UP:
            loss1 = self.loss_func(outputs, target_a)
            loss2 = self.loss_func(outputs, target_b)
            loss = lam * loss1 + (1 - lam) * loss2
        elif self.cfg.INPUT.USE_CUT_MIX and self.use_cut_mix:
            loss1 = self.loss_func(outputs, target_a_var)
            loss2 = self.loss_func(outputs, target_b_var)
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            loss = self.loss_func(outputs, target)

        if self.current_iteration % self.cfg.SOLVER.TENSORBOARD.LOG_PERIOD == 0:
            if self.summary_writer:
                self.summary_writer.add_scalar('Train/loss', loss, self.current_iteration)
        loss.backward()
        self.optim.step()

        if type(outputs) == type(()) and len(outputs) > 1:
            _output = outputs[0]
            for output in outputs[1:]:
                _output = _output + output
            outputs = _output / len(outputs)

        target = target.data.cpu()
        outputs = outputs.data.cpu()

        f1, acc = calculate_score(self.cfg, outputs, target)


        self.loss_avg.update(loss.cpu().item())
        self.acc_avg.update(acc)
        self.f1_avg.update(f1)

        return self.loss_avg.avg, self.acc_avg.avg, self.f1_avg.avg

    def evaluate(self):
        self.model.eval()
        print(len(self.val_dl))

        with torch.no_grad():

            all_outputs = list()
            all_targets = list()

            for batch in tqdm(self.val_dl, total=len(self.val_dl),
                              leave=False):
                data, target = batch
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)
                loss = self.loss_func(outputs, target)
                if type(outputs) == type(()) and len(outputs) > 1:
                    _output = outputs[0]
                    for output in outputs:
                        _output = _output + output
                    outputs = _output / len(outputs)
                target = target.data.cpu()
                outputs = outputs.data.cpu()

                self.val_loss_avg.update(loss.cpu().item())

                all_outputs.append(outputs)
                all_targets.append(target)

            all_outputs = torch.cat(all_outputs, 0)
            all_targets = torch.cat(all_targets, 0)

        val_f1, val_acc = calculate_score(self.cfg, all_outputs, all_targets)

        self.logger.info('Validation Result:')

        self.logger.info('VAL_LOSS: %s, VAL_ACC: %s VAL_F1: %s \n' % (self.val_loss_avg.avg, val_acc, val_f1))

        self.logger.info('-' * 20)

        if self.summary_writer:

            self.summary_writer.add_scalar('Valid/loss', self.val_loss_avg.avg, self.train_epoch)
            self.summary_writer.add_scalar('Valid/acc', np.mean(val_acc), self.train_epoch)
            self.summary_writer.add_scalar('Valid/f1',  np.mean(val_f1), self.train_epoch)

    def save(self):
        torch.save(self.model.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch' + str(self.train_epoch) + '.pth'))
        torch.save(self.optim.state_dict(), osp.join(self.output_dir,
                                                     self.cfg.MODEL.NAME + '_epoch' + str(
                                                         self.train_epoch) + '_optim.pth'))