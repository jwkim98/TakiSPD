import tensorflow as tf
import os
from word2vec import gensim_word2vec as word_vec


def main():
    print(tf.__version__)
    get_gensim_model()


def get_gensim_model():
    print('Downloading pretrained model')
    model = word_vec.get_trained_model()
    example = model['computer']
    print('Checking... computer : ' + example)
    model.save(os.path.join(os.path.curdir, 'word2vec', 'TrainedModel'))


if __name__ == '__main__':
    main()
