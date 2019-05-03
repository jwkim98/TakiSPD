import tensorflow as tf
import os
from word2vec import gensim_word2vec as word_vec
from gensim.models import KeyedVectors


def main():
    print(tf.__version__)
    load_gensim_model_example()


def get_gensim_model():
    print('Downloading pretrained model')
    model = word_vec.get_trained_model()
    model.wv.save(os.path.join(os.path.curdir, 'word2vec', 'TrainedModel', 'word2vec.kv'))


def load_gensim_model_example():
    loaded_model = KeyedVectors.load(os.path.join(os.path.curdir, 'word2vec', 'TrainedModel', 'word2vec.kv'))
    vector = loaded_model['sale']
    print(vector)


if __name__ == '__main__':
    main()
