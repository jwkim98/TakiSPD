import tensorflow as tf
from tensorflow.contrib import rnn


class Model:
    def __init__(self, batch_size, time_length, input_size, output_size, hidden_size):
        self.batch_size = batch_size
        self.time_length = time_length
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.weight = tf.Variable(tf.random_normal([hidden_size, output_size]))
        self.bias = tf.Variable(tf.random_normal([output_size]))

        self.final_output = 0
        self.final_state = 0

        self.model_input = 0
        self.model_label = 0
        self.softmax_output = 0

    def create_model(self):
        self.model_input = tf.placeholder(tf.float32, [self.batch_size, self.time_length, self.input_size],
                                          name='model_input')
        self.model_label = tf.placeholder(tf.float32, [self.batch_size, self.output_size],
                                          name='model_label')

        unrolled_input = tf.unstack(self.model_input, self.time_length, axis=1, name='unstack')
        cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0, state_is_tuple=True)

        output_list = []
        state = cell.zero_state(self.batch_size, tf.float32)
        for input_elem in unrolled_input:
            output, state = cell(input_elem, state)
            output_list.append(output)

        last_output = output_list[-1]

        self.final_state = state
        self.final_output = tf.matmul(last_output, self.weight) + self.bias

        self.softmax_output = tf.nn.softmax(self.final_output)

        return self.final_output

    def loss(self):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.model_label,
                                                                         logits=self.final_output))
        # if print_loss:
        #     tf.print(loss, [loss])
        return loss

    def train_accuracy(self):
        model_answer = tf.argmax(self.final_output, 1)
        real_answer = tf.argmax(self.model_label, 1)
        # equality = tf.equal(model_answer, real_answer)
        acc, acc_op = tf.metrics.accuracy(labels=real_answer, predictions=model_answer, name="train_metric")

        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train_metric")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        return running_vars_initializer, acc, acc_op

    def test_accuracy(self):
        model_answer = tf.argmax(self.final_output, 1)
        real_answer = tf.argmax(self.model_label, 1)
        # equality = tf.equal(model_answer, real_answer)
        acc, acc_op = tf.metrics.accuracy(labels=real_answer, predictions=model_answer, name="test_metric")

        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="test_metric")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        return running_vars_initializer, acc, acc_op

    def optimizer(self, loss):
        optimize = tf.train.AdamOptimizer()
        train = optimize.minimize(loss)
        return train








