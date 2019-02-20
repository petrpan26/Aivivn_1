from scripts.util import read_file, tokenize, make_embedding, text_to_sequences
from scripts.rnn import RNNKeras
from scripts.constant import DEFAULT_MAX_FEATURES
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scripts.rnn import RNNKeras
import argparse
import numpy as np
import datetime


def train_model(model, embedding_path, max_features=DEFAULT_MAX_FEATURES):
    model_name = '-'.join(
        '.'.join(str(datetime.datetime.now()).split('.')[:-1]).split(' '))

    data = read_file('./data/train.crash')
    tokenized_texts = tokenize(data['text'])
    labels = data['label'].values.astype(np.float16).reshape(-1, 1)

    embed_size, word_map, embedding_mat = make_embedding(
        tokenized_texts,
        embedding_path,
        max_features
    )

    texts_id = text_to_sequences(tokenized_texts, word_map)
    print(labels.shape)
    print(texts_id.shape)

    texts_id_train, texts_id_val, labels_train, labels_val = train_test_split(
        texts_id, labels, test_size=0.1)

    checkpoint = ModelCheckpoint(
        filepath='./Weights/{}-version.hdf5'.format(model_name),
        monitor='val_acc', verbose=1,
        mode='min',
        save_best_only=True
    )
    early = EarlyStopping(monitor='val_acc', mode='min', patience=3)
    callbacks_list = [checkpoint, early]
    batch_size = 16
    epochs = 100

    model = model(
        embeddingMatrix=embedding_mat,
        embed_size=embed_size,
        max_features=len(embedding_mat)
    )
    model.fit(
        texts_id_train, labels_train,
        validation_data=(texts_id_val, labels_val),
        callbacks=callbacks_list,
        epochs=epochs,
        batch_size=16
    )


model_dict = {
    'RNNKeras': RNNKeras
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        help='Model use',
        default='RNNKeras'
    )
    parser.add_argument(
        '-e',
        '--embedding',
        help='Model use',
        default='./embeddings/smallFasttext.vi.vec'
    )
    args = parser.parse_args()
    if not args.model in model_dict:
        raise RuntimeError('Model not found')
    train_model(model_dict[args.model], args.embedding)
