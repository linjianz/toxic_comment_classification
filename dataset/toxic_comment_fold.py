#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-16 12:34:33
Program: 
Description: 
"""

from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
import pandas as pd
from utils.embedding_utils import tokenize_sentences, read_embedding_list, clear_embedding_list, convert_tokens_to_ids


class ToxicComment(object):
    def __init__(self, dir_data, sentence_length, fold_count):
        self.dir_data = dir_data
        self.sentence_length = sentence_length
        self.fold_count = fold_count
        self.dir_save = self.dir_data + '/train-test-len-{:d}'.format(self.sentence_length)
        self.dir_train = self.dir_data + '/train.csv'
        self.dir_test = self.dir_data + '/test.csv'
        self.dir_embedding = self.dir_data + '/crawl-300d-2M.vec'
        self.UNKNOWN_WORD = '_UNK_'
        self.NAN_WORD = '_NAN_'
        self.END_WORD = '_END_'
        self.CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        self.x_train, self.y_train, self.x_test, self.test_id, self.embeddings = self.load_data()

    def load_data(self):
        if os.path.exists(self.dir_save):
            print('Find pre-processed data\nLoading train.txt...')
            data_train = np.loadtxt(self.dir_save + '/train.txt')

            print('Loading label.txt...')
            label_train = np.loadtxt(self.dir_save + '/label.txt')

            print('Loading test.txt...')
            data_test = np.loadtxt(self.dir_save + '/test.txt')

            print('Loading test id...')
            test_data = pd.read_csv(self.dir_test)
            test_id = test_data["id"].values
            test_id = test_id.reshape((len(test_id), 1))

            print('Loading embedding_list.txt...')
            embedding_list = np.loadtxt(self.dir_save + '/embedding_list.txt')
        else:
            train_data = pd.read_csv(self.dir_train)
            test_data = pd.read_csv(self.dir_test)
            test_id = test_data["id"].values
            test_id = test_id.reshape((len(test_id), 1))
            list_sentences_train = train_data["comment_text"].fillna(self.NAN_WORD).values
            list_sentences_test = test_data["comment_text"].fillna(self.NAN_WORD).values
            label_train = train_data[self.CLASSES].values  # (159571, 6)

            print("Tokenizing sentences in train set...")
            tokenized_sentences_train, words_dict = tokenize_sentences(list_sentences_train, {})  # 159571
            print("Tokenizing sentences in test set...")
            tokenized_sentences_test, words_dict = tokenize_sentences(list_sentences_test, words_dict)  # 153164
            words_dict[self.UNKNOWN_WORD] = len(words_dict)  # insert unknown_word to the last

            print("Loading embeddings...")
            embedding_list, embedding_word_dict = read_embedding_list(self.dir_embedding)
            embedding_size = len(embedding_list[0])  # 300

            print("Preparing data...")
            embedding_list, embedding_word_dict = clear_embedding_list(embedding_list, embedding_word_dict, words_dict)
            embedding_word_dict[self.UNKNOWN_WORD] = len(embedding_word_dict)
            embedding_list.append([0.] * embedding_size)
            embedding_word_dict[self.END_WORD] = len(embedding_word_dict)
            embedding_list.append([-1.] * embedding_size)

            id_to_word = dict((id, word) for word, id in words_dict.items())
            data_train = convert_tokens_to_ids(
                tokenized_sentences_train,
                id_to_word,
                embedding_word_dict,
                self.sentence_length)
            data_test = convert_tokens_to_ids(
                tokenized_sentences_test,
                id_to_word,
                embedding_word_dict,
                self.sentence_length)

            os.mkdir(self.dir_save)
            np.savetxt(self.dir_save + '/train.txt', data_train)
            np.savetxt(self.dir_save + '/test.txt', data_test)
            np.savetxt(self.dir_save + '/label.txt', label_train)
            np.savetxt(self.dir_save + '/embedding_list.txt', embedding_list)
            with open(self.dir_save + '/id_to_word.json', 'w') as f:
                json.dump(id_to_word, f)
            print('Data saved in {:s}'.format(self.dir_save))

        return data_train, label_train, data_test, test_id, embedding_list

    def get_fold_by_id(self, idx):
        fold_size = len(self.x_train) // self.fold_count
        fold_start = fold_size * idx
        fold_end = fold_start + fold_size
        if idx == fold_size - 1:
            fold_end = len(self.x_train)

        x_t = np.concatenate([self.x_train[:fold_start], self.x_train[fold_end:]])
        y_t = np.concatenate([self.y_train[:fold_start], self.y_train[fold_end:]])
        x_v = self.x_train[fold_start:fold_end]
        y_v = self.y_train[fold_start:fold_end]

        return x_t, y_t, x_v, y_v


class ToxicCommentDataSet(Dataset):
    """toxic comment data set"""

    def __init__(self, x, embeddings, y=None, phase='Train'):
        self.x = x
        self.embeddings = embeddings
        self.y = y
        self.phase = phase

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """ get one sample"""
        sample = dict()
        sample['sentence'] = self.embeddings[self.x[idx].astype(int)].astype(np.float32)
        if self.phase == 'Train' or self.phase == 'Valid':
            sample['label'] = self.y[idx].astype(np.float32)

        return sample


def main():
    # train
    dir_data = '/media/csc105/Data/dataset-jiange/kaggle/toxic-comment-classification'
    sen_len = 500
    # data_set = ToxicComment(dir_data=dir_data, sentence_length=sen_len)
    # loader = DataLoader(data_set, batch_size=16, shuffle=False, num_workers=4)
    # n_batch = int(len(data_set.x)//loader.batch_size)
    # for i_batch, sample_batch in enumerate(loader):
    #     print(i_batch, '/', n_batch, ':', sample_batch['sentence'].size(), sample_batch['label'].size())

    # test
    # data_set = ToxicCommentDataSet(dir_data=dir_data, sentence_length=250, phase='Test')
    # loader = DataLoader(data_set, batch_size=16, shuffle=False, num_workers=4)
    # n_batch = int(len(data_set.x) // loader.batch_size)
    # for i_batch, sample_batch in enumerate(loader):
    #     print(i_batch, '/', n_batch, ':', sample_batch['sentence'].size())

if __name__ == '__main__':
    main()
