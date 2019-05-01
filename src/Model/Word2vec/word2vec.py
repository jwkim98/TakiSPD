import tensorflow as tf
import numpy as np


def create_one_hot_dict(word_dict):
    one_hot_dict = dict()
    vector_size = word_dict.size()
    for key, val in word_dict.items():
        vector = np.zeros([vector_size, 1])
        vector[val] = 1
        one_hot_dict[key] = vector
    return one_hot_dict


def generate_data(sentence_list, one_hot_dict, window_size):
    labeled_list = list()
    for sentence in sentence_list:
        for index in range(window_size, sentence.size()):
            label = one_hot_dict.get(sentence[index])
            data_list = list()
            for i in range(0, window_size):
                one_hot_word = one_hot_dict.get(sentence[index - (i + 1)])
                data_list.append(one_hot_word)
            labeled_list.append((data_list, label))




