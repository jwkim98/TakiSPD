import tensorflow as tf
import os
from word2vec import gensim_word2vec as word_vec


def main():
    print(tf.__version__)
    path = os.path.join(os.path.curdir, 'word2vec', 'TrainedModel')
    # word_vec.save_trained_model(path)
    load_gensim_model_example()


def load_gensim_model_example():
    filename = os.path.join(os.path.curdir, 'word2vec', 'TrainedModel', 'word2vec.kv')
    loaded_model = word_vec.load_key_vector(filename)
    vector = loaded_model['sale']
    print(vector.shape)


if __name__ == '__main__':
    main()
