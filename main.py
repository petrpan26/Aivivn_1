from scripts.util import read_file, tokenize, make_embedding, text_to_sequences
import numpy as np
from scripts.rnn import RNNKeras
from scripts.constant import DEFAULT_MAX_FEATURES
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint








data = read_file("./data/train.crash")
tokenized_texts = tokenize(data["text"])
labels = data["label"].values.astype(np.float16).reshape(-1, 1)


word_map, embedding_mat = make_embedding(
    tokenized_texts,
    embedding_path = "./data/baomoi.model.bin",
    embed_size = 400
)

def text_to_sequences_test(texts, word_map, max_len = 100):
    texts_id = []
    for sentence in texts:
        sentence = [word_map[word.lower()] for word in sentence[:min(max_len, sentence.shape[0])]]
        if len(sentence) < max_len:
            sentence = np.pad(sentence, (0, max_len - len(sentence)), 'constant', constant_values = 0)
        texts_id.append(sentence)

    return np.array(texts_id)



texts_id = text_to_sequences_test(tokenized_texts, word_map)
print(labels.shape)
print(texts_id.shape)

texts_id_train, texts_id_val, labels_train, labels_val = train_test_split(texts_id, labels, test_size = 0.1)

checkpoint = ModelCheckpoint(
    filepath = "./Weights/model.hdf5",
    monitor = 'val_loss', verbose = 1,
    mode = 'min',
    save_best_only = True
)
early = EarlyStopping(monitor = "val_acc", mode = "min", patience = 3)
callbacks_list = [checkpoint, early]
batch_size = 16
epochs = 100


model = RNNKeras(
    embeddingMatrix = embedding_mat,
    embed_size = 400,
    max_features = DEFAULT_MAX_FEATURES + 1
)
model.fit(
    texts_id_train, labels_train,
    validation_data = (texts_id_val, labels_val),
    callbacks = callbacks_list,
    epochs = epochs,
    batch_size = 16
)
