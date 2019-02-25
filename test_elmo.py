from scripts.util import read_file, tokenize, make_embedding, text_to_sequences, find_threshold
import numpy as np
from scripts.constant import DEFAULT_MAX_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from elmoformanylangs import Embedder
import tensorflow as tf
import random as rn
import pandas as pd
import timeit



from keras.models import Model, load_model, model_from_json
from keras.utils import Sequence
from keras.layers import Dense, Embedding, Input, GRU, Bidirectional, GlobalMaxPool1D, Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

# np random seed:
np.random.seed(22)

# # Setting the seed for python random numbers
rn.seed(1254)
#
# # Setting the graph-level random seed.
tf.set_random_seed(89)

elmo_path = "./data/elmo/"


batch_size = 16
epochs = 100



elmo = Embedder(elmo_path, batch_size = batch_size)

def to_length(texts, length):
    def pad_func(vector, pad_width, iaxis, kwargs):
        str = kwargs.get('padder', '<unk>')
        vector[:pad_width[0]] = str
        vector[-pad_width[1]:] = str
        return vector

    ret = []
    for sentence in texts:
        sentence = np.array(sentence, dtype = np.unicode)
        sentence = sentence[:min(length, len(sentence))]
        if length > len(sentence):
            sentence = np.pad(
                sentence, mode = pad_func,
                pad_width = (0, length - len(sentence))
            )
        ret.append(sentence)

    return np.array(ret)


class TrainSeq(Sequence):
    def __init__(self, X, y, batch_size):
        self._X, self._y = X, y
        self._batch_size = batch_size

    def __len__(self):
        return len(self._X) // self._batch_size

    def __getitem__(self, idx):
        print(idx)
        print(self._X[idx * self._batch_size:(idx + 1) * self._batch_size].shape)
        # print(np.array(elmo.sents2elmo(self._X[idx * self._batch_size:(idx + 1) * self._batch_size])).shape)
        # print(self._y[idx * self._batch_size:(idx + 1) * self._batch_size].shape)
        return np.array(elmo.sents2elmo(self._X[idx * self._batch_size:(idx + 1) * self._batch_size])), \
               self._y[idx * self._batch_size:(idx + 1) * self._batch_size]


class TestSeq(Sequence):
    def __init__(self, x, batch_size):
        self._X = x
        self._batch_size = batch_size

    def __len__(self):
        return len(self._X) // batch_size

    def __getitem__(self, idx):
        return np.array(elmo.sents2elmo(self._X[idx * self._batch_size:(idx + 1) * self._batch_size]))




def RNNKerasCPUNoEmbedding(embed_size = 1024, maxlen = 100):
    inp = Input(shape = (maxlen, embed_size))
    x = Bidirectional(GRU(256, return_sequences = True))(inp)
    x = Dropout(0.5)(x)
    x = Bidirectional(GRU(256, return_sequences = True))(x)
    x = Dropout(0.5)(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation = "relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    return model





def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))





data = read_file("./data/train.crash")
data_test = read_file("./data/test.crash", is_train = False)

labels = data["label"].values.astype(np.float16).reshape(-1, 1)
texts = tokenize(data["text"])
texts_test = tokenize(data_test["text"])


texts = to_length(texts, 100)
texts_test = to_length(texts_test, 100)

texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts, labels,
    test_size = 0.05
)


checkpoint = ModelCheckpoint(
    filepath = "./Weights/model_elmo.hdf5",
    monitor = 'val_f1', verbose = 1,
    mode = 'max',
    save_best_only = True
)
early = EarlyStopping(monitor = "val_f1", mode = "max", patience = 3)
callbacks_list = [checkpoint, early]

train_seq = TrainSeq(texts_train, labels_train, batch_size = batch_size)
val_seq = TrainSeq(texts_val, labels_val, batch_size = batch_size)
test_seq = TestSeq(texts_test, batch_size = batch_size)


model = RNNKerasCPUNoEmbedding()
model.fit_generator(
    train_seq,
    validation_data = val_seq,
    callbacks = callbacks_list,
    epochs = epochs,
    workers = False
)




model.load_weights("./Weights/model_elmo.hdf5")
prediction_prob = model.predict_generator(val_seq, workers = False)

OPTIMAL_THRESHOLD = find_threshold(prediction_prob, labels_val)
print(OPTIMAL_THRESHOLD)
prediction = (prediction_prob > OPTIMAL_THRESHOLD).astype(np.int8)
print(f1_score(
    y_true = labels_val.reshape(-1),
    y_pred = prediction.reshape(-1)
))



prediction_test = model.predict_generator(test_seq, workers = False)
df_predicton = pd.read_csv("./data/sample_submission.csv")
df_predicton["label"] = (prediction_test > OPTIMAL_THRESHOLD).astype(np.int8)
print(df_predicton.shape[0])
df_predicton.to_csv("./prediction/prediction_elmo.csv", index = False)