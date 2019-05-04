import tensorflow as tf
import os
from word2vec import gensim_word2vec as word_vec
from data_formatter import read_csv as csv_reader


def main():
    print(tf.__version__)
    path = os.path.join(os.path.curdir, 'word2vec', 'TrainedModel')
    # word_vec.save_trained_model(path)
    # load_gensim_model_example()
    csv_path = os.path.join(os.path.curdir, '..', '..', 'Dataset', 'spam.csv')
    file, longest = csv_reader.read_csv(csv_path)
    print("longest length: %d total_size: %d" % (longest, len(file)))
    print("example : " + str(file[6]))


def load_gensim_model_example():
    filename = os.path.join(os.path.curdir, 'word2vec', 'TrainedModel', 'word2vec.kv')
    loaded_model = word_vec.load_key_vector(filename)
    vector = loaded_model['sale']
    print(vector.shape)


if __name__ == '__main__':
    main()
