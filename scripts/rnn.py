from keras.models import Model
from keras.layers import \
    Dense, Embedding, Input, \
    CuDNNGRU, GRU, LSTM, Bidirectional, CuDNNLSTM, \
    GlobalMaxPool1D, GlobalAveragePooling1D, Dropout, \
    Lambda, Concatenate, TimeDistributed, Layer
from .util import f1
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import keras.backend as K
from keras.activations import softmax
from keras_layer_normalization import LayerNormalization




def RNNKeras(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(CuDNNGRU(128, return_sequences = True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(CuDNNGRU(128, return_sequences = True))(x)
    x = Dropout(0.5)(x)

    max_pool = GlobalMaxPool1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    last = Lambda(lambda x: x[:, 0, :])(x)
    concat_pool = Concatenate(axis = -1)([last, max_pool, avg_pool])

    op = Dense(64, activation = "relu")(concat_pool)
    op = Dropout(0.5)(op)
    op = Dense(1, activation = "sigmoid")(op)

    model = Model(inputs = inp, outputs = op)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

def RNNKerasCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(GRU(128, return_sequences = True, recurrent_dropout = 0.5, dropout = 0.5))(x)
    # x = Dropout(0.5)(x)
    x = Bidirectional(GRU(128, return_sequences = True, recurrent_dropout = 0.5, dropout = 0.5))(x)
    # x = Dropout(0.5)(x)

    max_pool = GlobalMaxPool1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    last = Lambda(lambda x: x[:, 0, :])(x)
    concat_pool = Concatenate(axis = -1)([last, max_pool, avg_pool])

    op = Dense(64, activation = "relu")(concat_pool)
    op = Dropout(0.5)(op)
    op = Dense(1, activation = "sigmoid")(op)

    model = Model(inputs = inp, outputs = op)
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
        # attention_type = SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_regularizer_weight=1e-4,
    )(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = SeqWeightedAttention()(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation = "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

def SARNNKeras(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100, rnn_type = CuDNNLSTM):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(rnn_type(128, return_sequences = True))(x)
    x = SeqSelfAttention(
        # attention_type = SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_regularizer_weight=1e-4,
    )(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Bidirectional(rnn_type(128, return_sequences = True))(x)
    x = SeqWeightedAttention()(x)
    # x = LayerNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation = "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model


def HRNNCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, max_nb_sent = 3, max_sent_len = 40):
    sent_inp = Input(shape = (max_sent_len, ))
    embed = Embedding(
        input_dim = max_features,
        output_dim = embed_size,
        weights = [embeddingMatrix],
        trainable = True
    )(sent_inp)
    word_lstm = Bidirectional(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2))(embed)
    sent_encoder = Model(sent_inp, word_lstm)

    doc_input = Input(shape = (max_nb_sent, max_sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2))(doc_encoder)
    preds = Dense(1, activation = "sigmoid")(sent_lstm)
    model = Model(inputs = doc_input, outputs = preds)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

def HRNN(embeddingMatrix = None, embed_size = 400, max_features = 20000, max_nb_sent = 3, max_sent_len = 40):
    sent_inp = Input(shape = (max_sent_len, ))
    embed = Embedding(
        input_dim = max_features,
        output_dim = embed_size,
        weights = [embeddingMatrix],
        trainable = True
    )(sent_inp)
    word_lstm = Bidirectional(CuDNNLSTM(128))(embed)
    sent_encoder = Model(sent_inp, word_lstm)

    doc_input = Input(shape = (max_nb_sent, max_sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(CuDNNLSTM(128))(doc_encoder)
    preds = Dense(1, activation = "sigmoid")(sent_lstm)
    model = Model(inputs = doc_input, outputs = preds)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model


def OriginalHARNNCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, max_nb_sent = 3, max_sent_len = 40):
    sent_inp = Input(shape = (max_sent_len, ))
    embed = Embedding(
        input_dim = max_features,
        output_dim = embed_size,
        weights = [embeddingMatrix],
        trainable = True
    )(sent_inp)
    word_lstm = Bidirectional(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = True))(embed)
    word_att = AttLayer(context_size = 256)(word_lstm)
    sent_encoder = Model(sent_inp, word_att)

    doc_input = Input(shape = (max_nb_sent, max_sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = True))(doc_encoder)
    sent_att = AttLayer(context_size = 256)(sent_lstm)
    preds = Dense(1, activation = "sigmoid")(sent_att)
    model = Model(inputs = doc_input, outputs = preds)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model

def OriginalHARNN(embeddingMatrix = None, embed_size = 400, max_features = 20000, max_nb_sent = 3, max_sent_len = 40):
    sent_inp = Input(shape = (max_sent_len, ))
    embed = Embedding(
        input_dim = max_features,
        output_dim = embed_size,
        weights = [embeddingMatrix],
        trainable = True
    )(sent_inp)
    word_lstm = Bidirectional(CuDNNLSTM(128, return_sequences = True))(embed)
    word_att = AttLayer(context_size = 256)(word_lstm)
    word_att = Dropout(0.5)(word_att)
    sent_encoder = Model(sent_inp, word_att)

    doc_input = Input(shape = (max_nb_sent, max_sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(CuDNNLSTM(128, return_sequences = True))(doc_encoder)
    sent_att = AttLayer(context_size = 256)(sent_lstm)
    sent_att = Dropout(0.5)(sent_att)
    preds = Dense(1, activation = "sigmoid")(sent_att)
    model = Model(inputs = doc_input, outputs = preds)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model


class AttLayer(Layer):
    def __init__(self, context_size):
        self._context_size = context_size
        self.supports_masking = True
        # self._linear = Dense(context_size, activation = "tanh")
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        self._W = self.add_weight(
            shape = (input_shape[-1], self._context_size),
            initializer="he_normal",
            trainable=True
        )
        self._b = self.add_weight(
            shape = (1, self._context_size),
            initializer="constant",
            trainable=True
        )
        self._context = self.add_weight(
            shape = (self._context_size, 1),
            initializer = "he_normal",
            trainable = True
        )
        super(AttLayer, self).build(input_shape)


    def compute_mask(self, input, input_mask=None):
        return input_mask


    def call(self, input, mask = None):
        # input: (N, T, M)
        rep = K.tanh(K.dot(input, self._W) + self._b) # (N, T, C)
        score = K.squeeze(K.dot(rep, self._context), axis = -1) # (N, T)

        weight = K.exp(score)
        if mask is not None:
            weight *= K.cast(mask, K.floatx())

        weight /= K.cast(K.sum(weight, axis = 1, keepdims = True) + K.epsilon(), K.floatx())


        # weight = softmax(score, axis = -1) # (N, T)
        op = K.batch_dot(input, weight, axes = (1, 1)) # (N, M)

        return op

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def HARNNCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, max_nb_sent = 3, max_sent_len = 40):
    sent_inp = Input(shape = (max_sent_len, ))
    embed = Embedding(
        input_dim = max_features,
        output_dim = embed_size,
        weights = [embeddingMatrix],
        trainable = True
    )(sent_inp)
    word_lstm = Bidirectional(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = True))(embed)
    word_att = SeqWeightedAttention()(word_lstm)
    sent_encoder = Model(sent_inp, word_att)

    doc_input = Input(shape = (max_nb_sent, max_sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, return_sequences = True))(doc_encoder)
    sent_att = SeqWeightedAttention()(sent_lstm)
    preds = Dense(1, activation = "sigmoid")(sent_att)
    model = Model(inputs = doc_input, outputs = preds)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model



def HARNN(embeddingMatrix = None, embed_size = 400, max_features = 20000, max_nb_sent = 3, max_sent_len = 40):
    sent_inp = Input(shape = (max_sent_len, ))
    embed = Embedding(
        input_dim = max_features,
        output_dim = embed_size,
        weights = [embeddingMatrix],
        trainable = True
    )(sent_inp)
    word_lstm = Bidirectional(CuDNNLSTM(128, return_sequences = True))(embed)
    word_att = SeqWeightedAttention()(word_lstm)
    word_att = Dropout(0.5)(word_att)
    sent_encoder = Model(sent_inp, word_att)

    doc_input = Input(shape = (max_nb_sent, max_sent_len))
    doc_encoder = TimeDistributed(sent_encoder)(doc_input)
    sent_lstm = Bidirectional(CuDNNLSTM(128, return_sequences = True))(doc_encoder)
    sent_att = SeqWeightedAttention()(sent_lstm)
    sent_att = Dropout(0.5)(sent_att)
    preds = Dense(1, activation = "sigmoid")(sent_att)
    model = Model(inputs = doc_input, outputs = preds)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model



