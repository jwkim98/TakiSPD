import tensorflow as tf
import numpy as np
import random
import math


# Converts word dictionary (word, number) to dictionary of (word, one_hot_vector)
def create_one_hot_dict(word_dict):
    one_hot_dict = dict()
    vector_size = word_dict.size()
    for key, val in word_dict.items():
        vector = np.zeros([vector_size, 1])
        vector[val] = 1
        one_hot_dict[key] = vector
    return one_hot_dict


# Generates labeled data set in tuple form for training and testing
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

    random.shuffle(labeled_list)
    return labeled_list


def create_model(one_hot_size, embedding_dimension):
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[None, 2, one_hot_size])
        train_labels = tf.placeholder(tf.int32, shape=[None, one_hot_size])

        with tf.name_scope('matrices'):
            to_hidden = tf.Variable(tf.truncated_normal([one_hot_size, embedding_dimension],
                              stddev=1.0 / math.sqrt(embedding_dimension)))

            to_result = tf.Variable(tf.truncated_normal([embedding_dimension, one_hot_size],
                              stddev=1.0 / math.sqrt(one_hot_size)))

        #bias = tf.Variable(tf.zeros([embedding_dimension, 1]))
        #bias = tf.transpose(bias)

        with tf.name_scope('mul'):
            hidden = tf.matmul(train_inputs, to_hidden)
            result = tf.matmul(hidden, to_result)

        with tf.name_scope('optimize'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=result, labels=train_labels))
            train_step = tf.train.AdamOptimizer(2e-5).minimize(loss, name='train')






