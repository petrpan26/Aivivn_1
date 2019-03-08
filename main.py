from scripts.util import read_file, tokenize, make_embedding, text_to_sequences
from scripts.rnn import RNNKeras
from scripts.constant import DEFAULT_MAX_FEATURES
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.rnn import RNNKeras, RNNKerasCPU, LSTMKeras, SARNNKerasCPU, SARNNKeras
from scripts.cnn import TextCNN
import argparse
import os
import numpy as np
import datetime
import pandas as pd
from scripts.util import find_threshold
from sklearn.metrics import f1_score


def train_model(model, embedding_path, max_features, should_find_threshold, should_mix):
    model_name = '-'.join(
        '.'.join(str(datetime.datetime.now()).split('.')[:-1]).split(' '))

    train_data = read_file('./data/train.crash')
    test_data = read_file('./data/test.crash', is_train=False)
    train_tokenized_texts = tokenize(train_data['text'])
    test_tokenizes_texts = tokenize(test_data['text'])
    labels = train_data['label'].values.astype(np.float16).reshape(-1, 1)

    embed_size, word_map, embedding_mat = make_embedding(
        list(train_tokenized_texts) +
        list(test_tokenizes_texts) if should_mix else train_tokenized_texts,
        embedding_path,
        max_features
    )

    texts_id = text_to_sequences(train_tokenized_texts, word_map)
    print('Number of train data: {}'.format(labels.shape))

    texts_id_train, texts_id_val, labels_train, labels_val = train_test_split(
        texts_id, labels, test_size=0.05)

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
    early = EarlyStopping(monitor='val_f1', mode='max', patience=3)
    callbacks_list = [checkpoint, early]
    batch_size = 16
    epochs = 100

    model = model(
        embeddingMatrix=embedding_mat,
        embed_size=embed_size,
        max_features=embedding_mat.shape[0]
    )
    model.fit(
        texts_id_train, labels_train,
        validation_data=(texts_id_val, labels_val),
        callbacks=callbacks_list,
        epochs=epochs,
        batch_size=batch_size
    )

    model.load_weights('{}/models.hdf5'.format(model_path))
    prediction_prob = model.predict(texts_id_val)
    if should_find_threshold:
        OPTIMAL_THRESHOLD = find_threshold(prediction_prob, labels_val)
    else:
        OPTIMAL_THRESHOLD = 0.5
    print('OPTIMAL_THRESHOLD: {}'.format(OPTIMAL_THRESHOLD))
    prediction = (prediction_prob > OPTIMAL_THRESHOLD).astype(np.int8)
    print('F1 validation score: {}'.format(f1_score(prediction, labels_val)))
    with open('{}/f1'.format(model_path), 'w') as fp:
        fp.write(str(f1_score(prediction, labels_val)))

    test_id_texts = text_to_sequences(test_tokenizes_texts, word_map)
    test_prediction = model.predict(test_id_texts)

    df_predicton = pd.read_csv("./data/sample_submission.csv")
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
    'TextCNN': TextCNN
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
        '--max',
        help='Model use',
        default=DEFAULT_MAX_FEATURES
    )
    parser.add_argument(
        '--find_threshold',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--mix',
        action='store_true',
        help='Model use'
    )
    args = parser.parse_args()
    if not args.model in model_dict:
        raise RuntimeError('Model not found')
    train_model(model_dict[args.model], args.embedding,
                int(args.max), args.mix)
