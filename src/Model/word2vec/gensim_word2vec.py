import os
import sys
import gensim
import subprocess
from keras.utils import get_file
import numpy as np


# Download pre-trained gensim model
def get_trained_model():
    MODEL = 'GoogleNews-vectors-negative300.bin'

    path = get_file(MODEL + '.gz', 'https://deeplearning4jblob.blob.core.windows.net/resources/wordvectors/%s.gz' % MODEL)
    unzipped = os.path.join('generated', MODEL)
    if not os.path.isfile(unzipped):
        with open(unzipped, 'wb') as fout:
            # Creates new process called 'zcat' and reads from path(stdin) and rites to 'wb'(stdout)
            zcat = subprocess.Popen(['zcat'], stdin=open(path), stdout=fout, stderr=sys.stderr)
            zcat.wait()

    model = gensim.models.KeyedVectors.load_word2vec_format(unzipped, binary=True)
    return model


def get_word_vector(model, word):
    return model.wv[word]


