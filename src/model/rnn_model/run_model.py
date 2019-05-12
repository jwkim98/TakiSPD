import tensorflow as tf
from rnn_model import model as spd
import numpy as np
import random
import os
import matplotlib.pyplot as  plt


def get_model():
    model = spd.Model(batch_size=100, time_length=139, input_size=300, output_size=2, hidden_size=100)
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

        model_label = np.array([1, 0])
        if label == True:
            model_label = np.array([1, 0])
        else:
            model_label = np.array([0, 1])

        result_list.append((np.array(word_vector_list), model_label))

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

    loss = model.loss()
    optimize = model.optimizer(loss)
    train_metric_initializer, train_acc, train_acc_op = model.train_accuracy()
    test_metric_initializer, test_acc, test_acc_op = model.test_accuracy()

    saver = tf.train.Saver()
    path = os.path.join(os.path.curdir, 'saved_model')

    with tf.Session() as sess:
        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        sess.run(init_global)
        sess.run(init_local)

        test_acc_list = list()
        train_acc_list = list()
        epoch_label_list = list()
        for i in range(0, epochs):
            sess.run(test_metric_initializer)
            sess.run(train_metric_initializer)

            train_batch = get_batch(train_vector_list, batch_size)
            test_batch = get_batch(test_vector_list, batch_size)

            train_data_input = np.array([data_tuple[0] for data_tuple in train_batch])
            train_label_input = np.array([data_tuple[1] for data_tuple in train_batch])

            # print(np.shape(train_data_input))
            # print(np.shape(train_label_input))

            test_data_input = np.array([test_tuple[0] for test_tuple in test_batch])
            test_label_input = np.array([test_tuple[1] for test_tuple in test_batch])

            # print("running")
            sess.run(optimize, feed_dict={model.model_input: train_data_input, model.model_label: train_label_input})

            sess.run(train_acc_op, feed_dict={model.model_input: train_data_input, model.model_label: train_label_input})
            acc_train = sess.run(train_acc)

            sess.run(test_acc_op, feed_dict={model.model_input: test_data_input, model.model_label: test_label_input})
            acc_test = sess.run(test_acc)

            # softmax_out = sess.run(model.softmax_output,
            #          feed_dict={model.model_input: train_data_input, model.model_label: train_label_input})
            #
            # label_out = sess.run(model.model_label, feed_dict={model.model_input: test_data_input, model.model_label: test_label_input})


            # print("softmax_output: " + str(softmax_out))
            # print("label_output: " + str(label_out))

            if i % 10 == 0:
                train_acc_list.append(acc_train*100)
                test_acc_list.append(acc_test*100)
                epoch_label_list.append(i)
                print("train_acc: " + str(acc_train))
                print("test_acc: " + str(acc_test))
                print("Epochs: " + str(i))
            if i % 100 == 0:
                name = 'model_' + str(i/10) + '.ckpt'
                saver.save(sess, os.path.join(path, name))

        save_path = saver.save(sess, os.path.join(path, 'model_final.ckpt'))
        print("saved model to %s: " % save_path)

        plt.plot(np.array(epoch_label_list), np.array(train_acc_list), 'ro', label='train'
                , markersize = 0.3)
        plt.plot(np.array(epoch_label_list), np.array(test_acc_list), 'bo', label='test'
                , markersize = 0.3)
        # print(epoch_label_list)
        # print(train_acc_list)
        # print(test_acc_list)

        plt.axis([0, epochs, 50, 100])
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc='upper left')
        plt.show()
        plt.savefig('Analysis.png')

        # inputs = {
        #     'model_input': model.model_input,
        #     'model_label': model.model_label
        # }
        #
        # outputs = {'final_output' : model.softmax_output}
        # tf.saved_model.simple_save(
        #     sess, path + '/', inputs, outputs)
