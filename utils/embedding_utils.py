#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-06 13:55:21
Program: 
Description: 
"""
import nltk
import numpy as np
from tqdm import tqdm


def tokenize_sentences(sentences, words_dict):
    """
    :param sentences: list
    :param words_dict: {}
    :return:
    tokenized_sentences: list
    words_dict: {'word': word_id}
    """
    tokenized_sentences = []
    for sentence in tqdm(sentences):
        if hasattr(sentence, "decode"):
            sentence = sentence.decode("utf-8")
        tokens = nltk.tokenize.word_tokenize(sentence)
        result = []
        for word in tokens:
            word = word.lower()
            if word not in words_dict:
                words_dict[word] = len(words_dict)
            word_index = words_dict[word]
            result.append(word_index)
        tokenized_sentences.append(result)
    return tokenized_sentences, words_dict


def read_embedding_list(file_path):
    """
    return:
        embedding_list:         2M x 300
        embedding_word_dict:    {'word': id} length 2M
    """
    embedding_word_dict = {}
    embedding_list = []
    with open(file_path) as f:
        for row in tqdm(f.read().split("\n")[1:-1]):
            data = row.split(" ")
            word = data[0]
            embedding = np.array([float(num) for num in data[1:-1]])
            embedding_list.append(embedding)
            embedding_word_dict[word] = len(embedding_word_dict)

    embedding_list = np.array(embedding_list)
    return embedding_list, embedding_word_dict


def clear_embedding_list(embedding_list, embedding_word_dict, words_dict):
    """
    return:
        cleared_embedding_list:         W x 300, W is the number --> words_dict & embedding_word_dict
        cleared_embedding_word_dict:    {'word': id} length W
    """
    cleared_embedding_list = []
    cleared_embedding_word_dict = {}

    for word in words_dict:
        if word not in embedding_word_dict:
            continue
        word_id = embedding_word_dict[word]
        row = embedding_list[word_id]
        cleared_embedding_list.append(row)
        cleared_embedding_word_dict[word] = len(cleared_embedding_word_dict)

    return cleared_embedding_list, cleared_embedding_word_dict


def convert_tokens_to_ids(tokenized_sentences, words_list, embedding_word_dict, sentences_length):
    words_train = []

    for sentence in tokenized_sentences:
        current_words = []
        for word_index in sentence:  # eg: [10, 13, 8, ...]
            word = words_list[word_index]  # eg: 'the'
            word_id = embedding_word_dict.get(word, len(embedding_word_dict) - 2)  # id in embedding_word_dict
            current_words.append(word_id)

        if len(current_words) >= sentences_length:
            current_words = current_words[:sentences_length]
        else:
            current_words += [len(embedding_word_dict) - 1] * (sentences_length - len(current_words))  # add END_WORD
        words_train.append(current_words)
    return words_train
