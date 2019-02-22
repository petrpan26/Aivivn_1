import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Embedding, Input, CuDNNGRU, GRU, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers
import keras.backend as K
from .util import f1


def RNNKeras(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(CuDNNGRU(50, return_sequences = True))(x)
    # x = Dropout(0.1)(x)
    x = Bidirectional(CuDNNGRU(50, return_sequences = True))(x)
    x = Dropout(0.1)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

def RNNKerasCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(GRU(128, return_sequences = True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(128, return_sequences = True))(x)
    x = Dropout(0.5)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation = "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model