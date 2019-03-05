from keras.models import Model
from keras.layers import Dense, Embedding, Input, CuDNNGRU, GRU, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNLSTM
from .util import f1
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention


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

def LSTMKeras(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(CuDNNLSTM(50, return_sequences = True))(x)
    # x = Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(50, return_sequences = True))(x)
    x = Dropout(0.1)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation = "relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model


def SARNNKerasCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = SeqSelfAttention(
        attention_type = SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_regularizer_weight=1e-4,
    )(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = SeqWeightedAttention()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation = "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

def SARNNKeras(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences = True))(x)
    x = SeqSelfAttention(
        attention_type = SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_regularizer_weight=1e-4,
    )(x)
    # x = Dropout(0.5)(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences = True))(x)
    x = SeqWeightedAttention()(x)
    # x = Dropout(0.5)(x)
    x = Dense(64, activation = "relu")(x)
    # x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model
