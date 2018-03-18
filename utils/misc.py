#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-01 16:47:20
Program: 
Description: 
"""

import torch
import shutil
import os
import math
import numpy as np
from torch.autograd import Variable
from torch.nn import init


def pre_create_file_train(dir_model, dir_log, args):
    """
    :param dir_model: ./model
    :param dir_log:  ./log
    :param args:
    :return:
    mkdir ./model/lstm/20180101     if not exist, mkdir
    mkdir ./log/lstm/20180101       if exist, remove then mkdir
    """
    if not os.path.exists(dir_model):
        os.mkdir(dir_model)
    if not os.path.exists(dir_log):
        os.mkdir(dir_log)

    dir_models = dir_model + '/' + args.net_name
    dir_logs = dir_log + '/' + args.net_name
    dir_model_date = dir_models + '/' + args.dir_date
    dir_log_date = dir_logs + '/' + args.dir_date
    if not os.path.exists(dir_models):
        os.mkdir(dir_models)
    if not os.path.exists(dir_logs):
        os.mkdir(dir_logs)
    if not os.path.exists(dir_model_date):
        os.mkdir(dir_model_date)
    if os.path.exists(dir_log_date):
        shutil.rmtree(dir_log_date)
    os.mkdir(dir_log_date)
    return dir_model_date, dir_log_date


def pre_create_file_test(args):
    dir_test = './test'
    if not os.path.exists(dir_test):
        os.mkdir(dir_test)

    dir_net = dir_test + '/' + args.net_restore
    dir_time = dir_net + '/' + args.date_restore + '_' + args.model_restore
    if not os.path.exists(dir_net):
        os.mkdir(dir_net)
    if not os.path.exists(dir_time):
        os.mkdir(dir_time)
    return dir_time


def to_var(x):
    if torch.cuda.is_available():
        return Variable(x).cuda()
    else:
        return Variable(x)


def init_xavier(m):
    if isinstance(m, torch.nn.Conv2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0.0)
    if isinstance(m, torch.nn.Linear):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0.0)
    if isinstance(m, torch.nn.BatchNorm2d):
        init.xavier_normal(m.weight.data)
        init.constant(m.bias.data, 0.0)


def adjust_learning_rate(optimizer, epoch, lr_base, gamma=0.316, epoch_lr_decay=25):
    """
        epoch       lr
        000-025     1e-4
        025-050     3e-5
        050-075     1e-5
        075-100     3e-6
        100-125     1e-6
        125-150     3e-7
    """

    exp = int(math.floor(epoch / epoch_lr_decay))
    lr_decay = gamma ** exp
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_decay * lr_base


def display_loss(hour_per_epoch, epoch, args, step, step_per_epoch, optimizer, loss, loss_list, writer, step_global):
    """
    tensor board: loss, mean loss (reset every epoch)
    """
    loss_mean = np.mean(loss_list)
    print('\n{:.3f}h/E {:03d} [{:03d}/{:03d}] [lr {:.6f}] {:.3f} ({:.3f})'.format(
        hour_per_epoch, epoch + 1, step + 1, step_per_epoch, optimizer.param_groups[0]['lr'], loss,
        loss_mean))
    writer.add_scalars('./train',
                       {'loss_t': loss, 'loss_mean': loss_mean},
                       step_global)


def display_loss_tb_val(batch_v, loss_v, loss1_v, loss2_v, args, writer, step_global):
    print('\n{:d} batches: L {:.4f}={:.4f}+{:d}*{:.4f}'.format(batch_v, loss_v, loss1_v, args.beta, loss2_v))
    writer.add_scalars('./train-val', {'loss_v': loss_v, 'loss1_v': loss1_v, 'loss2_v': loss2_v}, step_global)
