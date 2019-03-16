from scripts.util import read_file, sent_tokenize, sent_embedding, text_sents_to_sequences
from scripts.constant import DEFAULT_MAX_FEATURES
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.rnn import HRNN, HRNNCPU, OriginalHARNN, OriginalHARNNCPU, HARNN, HARNNCPU
import argparse
import os
import numpy as np
import datetime
import pandas as pd
from scripts.util import find_threshold
from scripts.augment import shuffle_augment
from sklearn.metrics import f1_score


def train_model(
        model, embedding_path,
        max_features, max_nb_sent, max_sent_len,
        should_find_threshold, should_mix,
        return_prob, trainable, use_additive_emb, augment_size, aug_min_len
):
    model_name = '-'.join(
        '.'.join(str(datetime.datetime.now()).split('.')[:-1]).split(' '))

    train_data = read_file('./data/train.crash')
    test_data = read_file('./data/test.crash', is_train=False)
    train_tokenized_texts = sent_tokenize(train_data['text'])
    test_tokenizes_texts = sent_tokenize(test_data['text'])
    labels = train_data['label'].values.astype(np.float16).reshape(-1, 1)

    train_tokenized_texts, val_tokenized_texts, labels_train, labels_val = train_test_split(
        train_tokenized_texts, labels, test_size=0.05
    )

    augment_size = int(augment_size)
    aug_min_len = int(aug_min_len)
    max_nb_sent = int(max_nb_sent)
    max_sent_len = int(max_sent_len)

    if augment_size != 0:
        if augment_size < 0:
            augment_size = len(train_tokenized_texts) * (-augment_size)

        print(augment_size)

        train_tokenized_texts, labels_train = shuffle_augment(
            train_tokenized_texts,
            labels_train,
            n_increase = augment_size,
            min_length = aug_min_len
        )

    embed_size, word_map, embedding_mat = sent_embedding(
        list(train_tokenized_texts) + list(val_tokenized_texts) +
        list(test_tokenizes_texts) if should_mix
        else list(train_tokenized_texts) + list(val_tokenized_texts),
        embedding_path,
        max_features
    )

    texts_id_train = text_sents_to_sequences(
        train_tokenized_texts,
        word_map,
        max_nb_sent = max_nb_sent,
        max_sent_len = max_sent_len
    )

    texts_id_val = text_sents_to_sequences(
        val_tokenized_texts,
        word_map,
        max_nb_sent = max_nb_sent,
        max_sent_len = max_sent_len
    )


    # texts_id = text_sents_to_sequences(
    #     train_tokenized_texts,
    #     word_map,
    #     max_nb_sent = max_nb_sent,
    #     max_sent_len = max_sent_len
    # )
    print('Number of train data: {}'.format(labels.shape))

    # texts_id_train, texts_id_val, labels_train, labels_val = train_test_split(
    #     texts_id, labels, test_size=0.05)

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
    batch_size = 16
    epochs = 100

    model = model(
        embeddingMatrix=embedding_mat,
        embed_size=embed_size,
        max_features=embedding_mat.shape[0],
        max_nb_sent = max_nb_sent,
        max_sent_len = max_sent_len,
        trainable = trainable,
        use_additive_emb = use_additive_emb
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

    test_id_texts = text_sents_to_sequences(
        test_tokenizes_texts,
        word_map,
        max_nb_sent = max_nb_sent,
        max_sent_len = max_sent_len
    )
    test_prediction = model.predict(test_id_texts)

    df_predicton = pd.read_csv("./data/sample_submission.csv")

    if return_prob:
        df_predicton["label"] = test_prediction
    else:
        df_predicton["label"] = (
            test_prediction > OPTIMAL_THRESHOLD).astype(np.int8)
    print('Number of test data: {}'.format(df_predicton.shape[0]))
    df_predicton.to_csv('{}/prediction.csv'.format(model_path), index=False)


model_dict = {
    'HRNN': HRNN,
    'HRNNCPU': HRNNCPU,
    'HARNN': HARNN,
    'HARNNCPU': HARNNCPU,
    'OriginalHARNN': OriginalHARNN,
    'OriginalHARNNCPU':OriginalHARNNCPU
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        help='Model use',
        default='HRNN'
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
        '--nb_sent',
        help='Model use',
        default=3
    )
    parser.add_argument(
        '--sent_len',
        help='Model use',
        default=50
    )
    parser.add_argument(
        '--aug',
        help='Model use',
        default=0
    )
    parser.add_argument(
        '--aug_min_len',
        help='Model use',
        default=1
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
    parser.add_argument(
        '--prob',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--fix_embed',
        action='store_false',
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
    train_model(
        model_dict[args.model], args.embedding,
        int(args.max), args.nb_sent, args.sent_len,
        args.find_threshold, args.mix, args.prob,
        args.fix_embed, args.add_embed, args.aug, args.aug_min_len
    )
