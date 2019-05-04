import tensorflow as tf
import model as spd
import random


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
    with tf.Session() as sess:
        for i in range(0, epochs):
            train_batch = get_batch(train_vector_list, batch_size)
            test_batch = get_batch(test_vector_list, batch_size)
            sess.run(optimize)


