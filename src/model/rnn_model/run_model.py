import tensorflow as tf
import model as spd
import random
import os


def get_model():
    model = spd.model(batch_size=100, time_length=20, intput_size=300, output_size=2, hidden_size=100)
    model.create_model()
    return model


# receives spam_list of shape [number_of_spams, (word_list, label)] and
# converts them to list of word vector of shape [number_of_spams, (word_vector_list, label)]
def to_vector_list(key_vector, spam_list):

    result_list = []
    for word_list, label in spam_list:
        word_vector_list = []

        for word in word_list:
            vector = key_vector[word]
            word_vector_list.append(vector)

        result_list.append((word_vector_list, label))

    return result_list


# shuffles the list and brings batch of size batch_size
def get_batch(vector_list, batch_size):
    random.shuffle(vector_list)
    return vector_list[:batch_size]


def train(model, spam_list, key_vector, epochs):
    vector_list = to_vector_list(key_vector, spam_list)
    train_vector_list = vector_list[:90]
    test_vector_list = vector_list[90:]
    batch_size = 100

    loss = model.loss(print_loss=True, print_accuracy=True)
    optimize = model.optimizer(loss)

    saver = tf.train.Saver()
    path = os.path.join(os.path.curdir, 'saved_model')

    with tf.Session() as sess:
        for i in range(0, epochs):
            train_batch = get_batch(train_vector_list, batch_size)
            test_batch = get_batch(test_vector_list, batch_size)

            train_data_input = [data_tuple[0] for data_tuple in train_batch]
            train_label_input = [data_tuple[1] for data_tuple in train_batch]

            test_data_input = [test_tuple[0] for test_tuple in test_batch]
            test_label_input = [test_tuple[1] for test_tuple in test_batch]

            sess.run(optimize, feed_dict={model_input: train_data_input, label: train_label_input})
            sess.run(optimize, feed_dict={model_input: test_data_input, label: test_label_input})

            if i % 30 == 0:
                name = 'model_' + str(i/10) + '.ckpt'
                saver.save(sess, os.path.join(path, name))

        save_path = saver.save(sess, os.path.join(path, 'model_final.ckpt'))
        print("saved model to %s: " % save_path)
