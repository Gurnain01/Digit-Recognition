import pickle
import gzip

import numpy as np

def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')

    training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')

    f.close()

    return (training_data, validation_data, test_data)

def vectr_result(j):
    h = np.zeros((10, 1))
    h[j] = 1.0
    return h
