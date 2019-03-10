from scripts.util import \
    read_file, \
    tokenize, make_embedding, text_to_sequences, \
    sent_embedding, sent_tokenize, text_sents_to_sequences, f1
from scripts.constant import DEFAULT_MAX_FEATURES
from sklearn.model_selection import train_test_split
from scripts.rnn import SARNNKeras, HARNN, OriginalHARNN, AttLayer
from scripts.cnn import VDCNN, TextCNN, LSTMCNN
from scripts.stack import StackedGeneralizerWithHier
import argparse
import os
import numpy as np
import datetime
import pandas as pd
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from keras.utils import CustomObjectScope
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention



def stack(models_list, hier_models_list, embedding_path, max_features, should_mix):
    model_name = '-'.join(
        '.'.join(str(datetime.datetime.now()).split('.')[:-1]).split(' '))

    train_data = read_file('./data/train.crash')
    test_data = read_file('./data/test.crash', is_train=False)

    train_tokenized_texts = tokenize(train_data['text'])
    test_tokenizes_texts = tokenize(test_data['text'])

    train_tokenized_texts_sent = sent_tokenize(train_data['text'])
    test_tokenizes_texts_sent = sent_tokenize(test_data['text'])

    labels = train_data['label'].values.astype(np.float16).reshape(-1, 1)

    embed_size, word_map, embedding_mat = make_embedding(
        list(train_tokenized_texts) +
        list(test_tokenizes_texts) if should_mix else train_tokenized_texts,
        embedding_path,
        max_features
    )

    embed_size_sent, word_map_sent, embedding_mat_sent = sent_embedding(
        list(train_tokenized_texts_sent) +
        list(test_tokenizes_texts_sent) if should_mix else train_tokenized_texts_sent,
        embedding_path,
        max_features
    )


    texts_id = text_to_sequences(train_tokenized_texts, word_map)
    texts_id_sent = text_sents_to_sequences(
        train_tokenized_texts_sent,
        word_map_sent,
        max_nb_sent = 3,
        max_sent_len = 50
    )
    print('Number of train data: {}'.format(labels.shape))

    texts_id_train, texts_id_val, texts_id_sent_train, texts_id_sent_val, labels_train, labels_val = train_test_split(
        texts_id, texts_id_sent, labels, test_size=0.05)

    model_path = './models/{}-version'.format(model_name)

    try:
        os.mkdir('./models')
    except:
        print('Folder already created')
    try:
        os.mkdir(model_path)
    except:
        print('Folder already created')

    batch_size = 16
    epochs = 100
    patience = 3

    # meta_model = RandomForestClassifier (
    #     n_estimators=200,
    #     criterion="entropy",
    #     max_depth=5,
    #     max_features=0.5
    # )
    meta_model = MLPClassifier(
        hidden_layer_sizes = (10),
        early_stopping = True,
        validation_fraction = 0.05,
        batch_size = batch_size,
        n_iter_no_change = patience
    )
    models = [
        model(
            embeddingMatrix=embedding_mat,
            embed_size=embed_size,
            max_features=embedding_mat.shape[0]
        )
        for model in models_list
    ]

    hier_models = [
        model(
            embeddingMatrix=embedding_mat_sent,
            embed_size=embed_size_sent,
            max_features=embedding_mat_sent.shape[0],
            max_nb_sent = 3,
            max_sent_len = 50
        )
        for model in hier_models_list
    ]



    stack = StackedGeneralizerWithHier(models, hier_models, meta_model)
    stack.train_meta_model(
        X = texts_id_train, y = labels_train,
        X_val = texts_id_val, y_val = labels_val,
        X_hier = texts_id_sent_train, X_hier_val = texts_id_sent_val,
        model_path = model_path,
        epochs = epochs,
        batch_size = batch_size,
        patience = patience
    )

    stack.train_models(
        X = texts_id_train, y = labels_train,
        X_val = texts_id_val, y_val = labels_val,
        X_hier = texts_id_sent_train, X_hier_val = texts_id_sent_val,
        batch_size = batch_size,
        epochs = epochs,
        patience = patience,
        model_path = model_path
    )

    prediction = stack.predict(texts_id_val, texts_id_sent_val)
    print('F1 validation score: {}'.format(f1_score(prediction, labels_val)))
    with open('{}/f1'.format(model_path), 'w') as fp:
        fp.write(str(f1_score(prediction, labels_val)))

    test_id_texts = text_to_sequences(test_tokenizes_texts, word_map)
    test_id_texts_sent = text_sents_to_sequences(test_tokenizes_texts_sent, word_map_sent, 3, 50)
    test_prediction = stack.predict(test_id_texts, test_id_texts_sent)

    df_predicton = pd.read_csv("./data/sample_submission.csv")
    df_predicton["label"] = test_prediction

    print('Number of test data: {}'.format(df_predicton.shape[0]))
    df_predicton.to_csv('{}/prediction.csv'.format(model_path), index=False)



if __name__ == '__main__':
    models_list = [
        VDCNN, TextCNN, SARNNKeras
    ]
    hier_models_list = [
        HARNN
    ]
    parser = argparse.ArgumentParser()
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
        '--mix',
        action='store_true',
        help='Model use'
    )
    args = parser.parse_args()
    with CustomObjectScope({
        'SeqSelfAttention': SeqSelfAttention,
        'SeqWeightedAttention': SeqWeightedAttention,
        'AttLayer': AttLayer,
        'f1': f1}
    ):
        stack(models_list, hier_models_list, args.embedding,
                    int(args.max), args.mix)
