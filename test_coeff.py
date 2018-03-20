#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-20 16:27:14
Program: 
Description: 
"""
import argparse
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--coefficient',    default=1.4, type=float, help='normalize coefficient')
args = parser.parse_args()

print('\n')
print('START'.center(70, '='))
CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
print('Loading test id...')

dir_data = '/media/csc105/Data/dataset-jiange/kaggle/toxic-comment-classification'
dir_test = dir_data + '/test.csv'
dir_save = './model/lstm/20180317'

test_data = pd.read_csv(dir_test)
test_id = test_data["id"].values
test_id = test_id.reshape((len(test_id), 1))

predicts_list = list()
for i in range(10):
    predicts = np.loadtxt(dir_save + '/predicts-fold{}'.format(i))
    predicts_list.append(predicts)

predicts_ret = np.ones(predicts_list[0].shape)
for predicts_fold in predicts_list:
    predicts_ret *= predicts_fold

predicts_ret **= (1. / len(predicts_list))
predicts_ret **= args.coefficient

ret = pd.DataFrame(data=predicts_ret, columns=CLASSES)
ret['id'] = test_id
ret = ret[['id'] + CLASSES]
ret.to_csv(dir_save + '/submit-coeff-{:.1f}.csv'.format(args.coefficient), index=False)
print('DONE'.center(70, '='))
