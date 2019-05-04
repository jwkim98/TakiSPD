import tensorflow as tf
from rnn_model import model as spd
import numpy as np
import random
import os


def get_model():
    model = spd.Model(batch_size=100, time_length=139, input_size=300, output_size=1, hidden_size=100)
    model.create_model()
    return model


# receives spam_list of shape [number_of_spams, (word_list, label)] and
# converts them to list of word vector of shape [number_of_spams, (word_vector_list, label)]
def to_vector_list(key_vector, spam_list, longest):

    result_list = []
    for word_list, label in spam_list:
        word_vector_list = []

        for word in word_list:

            try:
                vector = key_vector[word]
            except:
                vector = np.zeros((300,), dtype=float)

            word_vector_list.append(np.array(vector))

        # Zero pad the word_vector_list for short vectors
        while len(word_vector_list) < longest:
            word_vector_list.append(np.zeros((300,), dtype=float))

        result_list.append((np.array(word_vector_list), np.array([label])))

    return result_list


# shuffles the list and brings batch of size batch_size
def get_batch(vector_list, batch_size):
    random.shuffle(vector_list)
    return vector_list[:batch_size]


# Trains the model with
def train(model, spam_list, longest_length, key_vector, epochs):
    vector_list = to_vector_list(key_vector, spam_list, longest_length)
    train_vector_list = vector_list[:4700]
    test_vector_list = vector_list[4700:]
    batch_size = 100

    loss = model.loss(print_loss=True)
    optimize = model.optimizer(loss)
    train_acc, train_acc_op = model.train_accuracy()
    test_acc, test_acc_op = model.test_accuracy()

    saver = tf.train.Saver()
    path = os.path.join(os.path.curdir, 'saved_model')

    with tf.Session() as sess:
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        sess.run(init_global)
        sess.run(init_local)
        for i in range(0, epochs):
            train_batch = get_batch(train_vector_list, batch_size)
            test_batch = get_batch(test_vector_list, batch_size)

            train_data_input = np.array([data_tuple[0] for data_tuple in train_batch])
            train_label_input = np.array([data_tuple[1] for data_tuple in train_batch])

            # print(np.shape(train_data_input))
            # print(np.shape(train_label_input))

            test_data_input = np.array([test_tuple[0] for test_tuple in test_batch])
            test_label_input = np.array([test_tuple[1] for test_tuple in test_batch])

            sess.run(optimize, feed_dict={model.model_input: train_data_input, model.model_label: train_label_input})

            acc_train = sess.run(train_acc,
                                 feed_dict={model.model_input: train_data_input, model.model_label: train_label_input})
            acc_test = sess.run(test_acc,
                                feed_dict={model.model_input: test_data_input, model.model_label: test_label_input})

            print("train_acc: " + str(acc_train))
            print("test_acc: " + str(acc_test))

            if i % 100 == 0:
                name = 'model_' + str(i/10) + '.ckpt'
                saver.save(sess, os.path.join(path, name))

        save_path = saver.save(sess, os.path.join(path, 'model_final.ckpt'))
        print("saved model to %s: " % save_path)