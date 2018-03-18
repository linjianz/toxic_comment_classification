#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-01 15:36:56
Program: 
Description: 
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import re
import os
import math
import argparse
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from dataset.toxic_comment import ToxicCommentDataSet
from utils.misc import to_var, adjust_learning_rate, pre_create_file_train, display_loss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server',         default=None, type=int, help='[6099]')
    parser.add_argument('--phase',          default=None, help='[Train/Test]')
    parser.add_argument('--sen_len',        default=None, type=int, help='sentence length')

    parser.add_argument('--net_name',       default=None, help='[lstm]')
    parser.add_argument('--dir_date',       default=None, help='Name it with date, such as 20180102')
    parser.add_argument('--batch_size',     default=32, type=int, help='Batch size')
    parser.add_argument('--lr_base',        default=1e-3, type=float, help='Base learning rate')
    parser.add_argument('--lr_decay_rate',  default=0.1, type=float, help='Decay rate of lr')
    parser.add_argument('--epoch_lr_decay', default=1000, type=int, help='Every # epoch, lr decay lr_decay_rate')

    parser.add_argument('--layer_num',      default=2, type=int, help='Lstm layer number')
    parser.add_argument('--hidden_size',    default=64, type=int, help='Lstm hidden units')
    parser.add_argument('--gpu',            default='0', help='GPU id list')
    parser.add_argument('--workers',        default=4, type=int, help='Workers number')

    return parser.parse_args()


def run_batch(sample, model, loss_func=None, optimizer=None, phase=None):
    """
    Run a batch for phase = {train, valid, test}
    """
    if phase == 'Train':
        model.train()
    else:
        model.eval()                            # test model，close dropout...

    x = to_var(sample['sentence'])
    label_pre = model(x)                        # [bs, 6]

    if phase == 'Train':
        label_gt = to_var(sample['label'])      # [bs, 6]
        loss = loss_func(label_pre, label_gt)
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # bp, compute gradients
        optimizer.step()                        # apply gradients
        return loss.data[0], label_pre.data
    else:
        return label_pre.data


def main(args):
    """
    1. Train till the loss minimized for 4 epoch
    2. Use the best model to test
    """
    print('\n\n')
    print('START'.center(70, '='))
    print('Net\t\t\t{:s}\nPhase\t\t\t{:s}\nSentence length\t\t{:d}'.format(args.net_name, args.phase, args.sen_len))
    torch.set_default_tensor_type('torch.FloatTensor')

    if args.phase == 'Train':
        model = Net(in_features=300, hidden_size=args.hidden_size, layer_num=args.layer_num, phase='Train')
        if torch.cuda.is_available():
            model = nn.DataParallel(model.cuda(), device_ids=args.gpu)

        print('LOADING DATA '.center(70, '='))
        dir_model_date, dir_log_date = pre_create_file_train(dir_model, dir_log, args)
        writer = SummaryWriter(dir_log_date)
        loss_func = nn.BCEWithLogitsLoss()  # loss(input, target)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base)
        data_set = ToxicCommentDataSet(dir_data=dir_data, sentence_length=args.sen_len, phase='Train')
        loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        step_per_epoch = int(math.ceil(len(data_set) / loader.batch_size))

        print('TRAIN'.center(70, '='))
        loss_best = -1
        epoch_best = 0
        epoch_current = 0
        while True:
            adjust_learning_rate(optimizer, epoch_current, args.lr_base, args.lr_decay_rate, args.epoch_lr_decay)
            loss_list = []
            for step, sample_batch in enumerate(loader):
                step_global = epoch_current * step_per_epoch + step
                tic = time()
                loss, _ = run_batch(sample=sample_batch,
                                    model=model,
                                    loss_func=loss_func,
                                    optimizer=optimizer,
                                    phase='Train')
                hour_per_epoch = step_per_epoch * ((time() - tic) / 3600)
                loss_list.append(loss)

                # display result and add to tensor board
                if (step + 1) % 10 == 0:
                    display_loss(hour_per_epoch, epoch_current, args, step, step_per_epoch, optimizer, loss, loss_list,
                                 writer, step_global)
            loss_mean = np.mean(loss_list)
            epoch_current += 1
            if loss_mean < loss_best or loss_best == -1:
                loss_best = loss_mean
                epoch_best = epoch_current
                torch.save(model.state_dict(), dir_model_date + '/model-best.pkl')
                print('>>>save current best model in {:s}\n'.format(dir_model_date + '/model-best.pkl'))
            else:
                if epoch_current - epoch_best == 5:
                    break

    print('TEST'.center(70, '='))
    model = Net(in_features=300, hidden_size=args.hidden_size, layer_num=args.layer_num, phase='Test')
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(), device_ids=args.gpu)
    dir_model_date = dir_model + '/' + args.net_name + '/' + args.dir_date
    model.load_state_dict(torch.load(dir_model_date + '/model-best.pkl'))
    print('>>>load best model in {:s}\n'.format(dir_model_date + '/model-best.pkl'))

    data_set = ToxicCommentDataSet(dir_data=dir_data, sentence_length=args.sen_len, phase='Test')
    loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    predicts = []  # 153164
    for step, sample_batch in enumerate(tqdm(loader)):
        predict = run_batch(sample=sample_batch, model=model, phase='Test')  # bs x 6
        predicts.extend(predict.cpu().numpy())
    ret = pd.DataFrame(data=predicts, columns=data_set.CLASSES)
    ret['id'] = data_set.test_id
    ret = ret[['id'] + data_set.CLASSES]
    ret.to_csv(dir_model_date+'/submit.csv', index=False)

    print('END'.center(70, '='))


if __name__ == '__main__':
    parser_args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu        # set visible gpu list, eg: '2,3,4'
    gpu_list = re.split('[, ]', parser_args.gpu)                # store the gpu id into a list
    parser_args.gpu = range(len(list(filter(None, gpu_list))))  # gpu for PyTorch

    if parser_args.server == 6099:
        dir_data = '/media/csc105/Data/dataset-jiange/kaggle/toxic-comment-classification'
        dir_project = '/home/jiange/project/toxic_comment_classification'
        dir_model = dir_project + '/model'      # directory to save model
        dir_log = dir_project + '/log'          # directory to save log
    else:
        raise Exception('Must give the right server id!')

    if parser_args.net_name == 'lstm':
        from net.lstm import Net
    elif parser_args.net_name == 'lstm_mean':
        from net.lstm_mean import Net
    elif parser_args.net_name == 'c_lstm':
        from net.c_lstm import Net
    elif parser_args.net_name == 'cnn':
        from net.cnn import Net
    else:
        raise Exception('Must give a net name')

    main(parser_args)
