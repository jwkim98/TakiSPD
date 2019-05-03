import os
import sys
import gensim
import subprocess
from keras.utils import get_file


# Download pre-trained gensim model
def save_trained_model(save_path):
    MODEL = 'GoogleNews-vectors-negative300.bin'
    # get trained model by google word2vec
    path = get_file(MODEL + '.gz', 'https://deeplearning4jblob.blob.core.windows.net/resources/wordvectors/%s.gz' % MODEL)
    unzipped = os.path.join(save_path,  MODEL)
    print(unzipped)
    if not os.path.isfile(unzipped):
        with open(unzipped, 'wb') as fout:
            # Creates new process called 'zcat' and reads from path(stdin) and rites to 'wb'(stdout)
            zcat = subprocess.Popen(['zcat'], stdin=open(path), stdout=fout, stderr=sys.stderr)
            zcat.wait()

    key_vector = gensim.models.KeyedVectors.load_word2vec_format(unzipped, binary=True)
    key_vector.wv.save(os.path.join(save_path, 'word2vec.kv'))
    return key_vector


def load_keyVector(filename):
    loaded_model = gensim.models.KeyedVectors.load(filename)
    return loaded_model


def get_word_vector(model, word):
    return model.wv[word]


