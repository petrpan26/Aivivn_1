from scripts.util import read_file, tokenize, make_embedding, text_to_sequences, find_threshold
import numpy as np
from scripts.constant import DEFAULT_MAX_FEATURES
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import random as rn
import pandas as pd



from keras.models import Model
from keras.layers import Dense, Embedding, Input, GRU, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention

# np random seed:
np.random.seed(22)

# # Setting the seed for python random numbers
rn.seed(1254)
#
# # Setting the graph-level random seed.
tf.set_random_seed(89)

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


def RNNKerasCPU(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100):
    inp = Input(shape = (maxlen, ))
    x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix])(inp)
    x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(128, return_sequences = True))(x)
    x = Dropout(0.5)(x)
    x = GlobalMaxPool1D()(x)
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
tokenized_texts = tokenize(data["text"])
labels = data["label"].values.astype(np.float16).reshape(-1, 1)

embed_size, word_map, embedding_mat = make_embedding(
    tokenized_texts,
    embedding_path = "./data/baomoi.model.bin",
    max_features =  40000
)



texts_id = text_to_sequences(tokenized_texts, word_map)
print(labels.shape)
print(texts_id.shape)

texts_id_train, texts_id_val, labels_train, labels_val = train_test_split(
    texts_id, labels,
    test_size = 0.05
)

checkpoint = ModelCheckpoint(
    filepath = "./Weights/model_sa_2.hdf5",
    monitor = 'val_f1', verbose = 1,
    mode = 'max',
    save_best_only = True
)
early = EarlyStopping(monitor = "val_f1", mode = "max", patience = 3)
callbacks_list = [checkpoint, early]
batch_size = 16
epochs = 100


model = SARNNKerasCPU(
    embeddingMatrix = embedding_mat,
    embed_size = 400,
    max_features = embedding_mat.shape[0]
)
model.fit(
    texts_id_train, labels_train,
    validation_data = (texts_id_val, labels_val),
    callbacks = callbacks_list,
    epochs = epochs,
    batch_size = 16
)




model.load_weights("./Weights/model_sa_2.hdf5")
prediction_prob = model.predict(texts_id_val)

OPTIMAL_THRESHOLD = find_threshold(prediction_prob, labels_val)
print(OPTIMAL_THRESHOLD)
prediction = (prediction_prob > OPTIMAL_THRESHOLD).astype(np.int8)
print(f1_score(
    y_true = labels_val.reshape(-1),
    y_pred = prediction.reshape(-1)
))



data_test = read_file("./data/test.crash", is_train = False)
tokenized_texts_test = tokenize(data_test["text"])
texts_id_test = text_to_sequences(tokenized_texts_test, word_map)
prediction_test = model.predict(texts_id_test)
df_predicton = pd.read_csv("./data/sample_submission.csv")
df_predicton["label"] = (prediction_test > OPTIMAL_THRESHOLD).astype(np.int8)
print(df_predicton.shape[0])
df_predicton.to_csv("./prediction/prediction_sa_2.csv", index = False)