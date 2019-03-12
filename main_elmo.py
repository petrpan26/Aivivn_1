from scripts.util import read_file, tokenize
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.rnn import RNNKeras, RNNKerasCPU, LSTMKeras, SARNNKerasCPU, SARNNKeras
from scripts.cnn import TextCNN, LSTMCNN, VDCNN
import argparse
import os
import numpy as np
import datetime
import pandas as pd
from scripts.util import find_threshold
from sklearn.metrics import f1_score
from keras.utils import Sequence
from elmoformanylangs import Embedder




def train_model(model, embedding_path, should_find_threshold, return_prob, use_additive_emb):
    batch_size = 16
    epochs = 100
    max_len = 100

    def to_length(texts, length):
        def pad_func(vector, pad_width, iaxis, kwargs):
            str = kwargs.get('padder', '<pad>')
            vector[:pad_width[0]] = str
            vector[-pad_width[1]:] = str
            return vector

        ret = []
        for sentence in texts:
            sentence = np.array(sentence, dtype=np.unicode)
            sentence = sentence[:min(length, len(sentence))]
            if length > len(sentence):
                sentence = np.pad(
                    sentence, mode=pad_func,
                    pad_width=(0, length - len(sentence))
                )
            ret.append(sentence)

        return np.array(ret)

    class TrainSeq(Sequence):
        def __init__(self, X, y, batch_size, elmo):
            self._X, self._y = X, y
            self._batch_size = batch_size
            self._indices = np.arange(len(self._X))
            self._elmo = elmo

        def __len__(self):
            return int(np.ceil(len(self._X) / float(self._batch_size)))

        def __getitem__(self, idx):
            id = self._indices[idx * self._batch_size:(idx + 1) * self._batch_size]
            return np.array(self._elmo.sents2elmo(self._X[id])), self._y[id]

        def on_epoch_end(self):
            np.random.shuffle(self._indices)

    class TestSeq(Sequence):
        def __init__(self, x, batch_size, elmo):
            self._X = x
            self._batch_size = batch_size
            self._elmo = elmo

        def __len__(self):
            return int(np.ceil(len(self._X) / float(self._batch_size)))

        def __getitem__(self, idx):
            return np.array(self._elmo.sents2elmo(self._X[idx * self._batch_size:(idx + 1) * self._batch_size]))

    model_name = '-'.join(
        '.'.join(str(datetime.datetime.now()).split('.')[:-1]).split(' '))

    elmo = Embedder(embedding_path, batch_size=batch_size)

    train_data = read_file('./data/train.crash')
    test_data = read_file('./data/test.crash', is_train=False)
    train_tokenized_texts = tokenize(train_data['text'])
    test_tokenizes_texts = tokenize(test_data['text'])
    labels = train_data['label'].values.astype(np.float16).reshape(-1, 1)

    texts = to_length(train_tokenized_texts, max_len)
    texts_test = to_length(test_tokenizes_texts, max_len)

    print('Number of train data: {}'.format(labels.shape))

    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, labels,
        test_size=0.05
    )

    model_path = './models/{}-version'.format(model_name)

    try:
        os.mkdir('./models')
    except:
        print('Folder already created')
    try:
        os.mkdir(model_path)
    except:
        print('Folder already created')

    checkpoint = ModelCheckpoint(
        filepath='{}/models.hdf5'.format(model_path),
        monitor='val_f1', verbose=1,
        mode='max',
        save_best_only=True
    )
    early = EarlyStopping(monitor='val_f1', mode='max', patience=5)
    callbacks_list = [checkpoint, early]

    train_seq = TrainSeq(texts_train, labels_train, batch_size=batch_size, elmo = elmo)
    val_seq = TrainSeq(texts_val, labels_val, batch_size=min(batch_size, len(texts_val)), elmo = elmo)
    test_seq = TestSeq(texts_test, batch_size=min(batch_size, len(texts_test)), elmo = elmo)

    model = model(
        maxlen = max_len,
        embed_size=1024,
        use_fasttext = True,
        use_additive_emb = use_additive_emb
    )
    model.fit_generator(
        train_seq,
        validation_data=val_seq,
        callbacks=callbacks_list,
        epochs=epochs,
        workers=False
    )

    model.load_weights('{}/models.hdf5'.format(model_path))
    prediction_prob = model.predict_generator(val_seq, workers=False)
    if should_find_threshold:
        OPTIMAL_THRESHOLD = find_threshold(prediction_prob, labels_val)
    else:
        OPTIMAL_THRESHOLD = 0.5
    print('OPTIMAL_THRESHOLD: {}'.format(OPTIMAL_THRESHOLD))
    prediction = (prediction_prob > OPTIMAL_THRESHOLD).astype(np.int8)
    print('F1 validation score: {}'.format(f1_score(prediction, labels_val)))
    with open('{}/f1'.format(model_path), 'w') as fp:
        fp.write(str(f1_score(prediction, labels_val)))

        test_prediction = model.predict_generator(test_seq, workers=False)

    df_predicton = pd.read_csv("./data/sample_submission.csv")
    if return_prob:
        df_predicton["label"] = test_prediction
    else:
        df_predicton["label"] = (
            test_prediction > OPTIMAL_THRESHOLD).astype(np.int8)

    print('Number of test data: {}'.format(df_predicton.shape[0]))
    df_predicton.to_csv('{}/prediction.csv'.format(model_path), index=False)


model_dict = {
    'RNNKeras': RNNKeras,
    'RNNKerasCPU': RNNKerasCPU,
    'LSTMKeras': LSTMKeras,
    'SARNNKerasCPU': SARNNKerasCPU,
    'SARNNKeras': SARNNKeras,
    'TextCNN': TextCNN,
    'LSTMCNN': LSTMCNN,
    'VDCNN': VDCNN
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        help='Model use',
        default='RNNKerasCPU'
    )
    parser.add_argument(
        '-e',
        '--embedding',
        help='Model use',
        default='./embeddings/smallFasttext.vi.vec'
    )
    parser.add_argument(
        '--find_threshold',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--prob',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--add_embed',
        action='store_true',
        help='Model use'
    )
    args = parser.parse_args()
    if not args.model in model_dict:
        raise RuntimeError('Model not found')
    train_model(model_dict[args.model], args.embedding, args.find_threshold, args.prob, args.add_embed)
