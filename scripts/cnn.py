from keras.models import Model
from keras.layers import \
    Dense, Embedding, Input, \
    Conv1D, MaxPool1D, \
    Dropout, \
    Lambda, Concatenate, Flatten
from .util import f1
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention



# Based on https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
def TextCNN(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100, filter_sizes = {2, 3, 4}):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)

    conv_ops = []
    for filter_size in filter_sizes:
        conv = Conv1D(128, filter_size, activation = 'relu')(x)
        pool = MaxPool1D(5)(conv)
        conv_ops.append(pool)

    concat = Concatenate(axis = 1)(conv_ops)
    concat = Dropout(0.5)(concat)
    flat = Flatten()(concat)

    op = Dense(64, activation = "relu")(flat)
    op = Dropout(0.5)(op)
    op = Dense(1, activation = "sigmoid")(op)

    model = Model(inputs = inp, outputs = op)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model
