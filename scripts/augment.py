import numpy as np
from gensim.models import KeyedVectors
import copy

def shuffle_augment(texts, labels, n_increase, min_length = 1):
    texts_long = []
    labels_long = []

    if min_length > 1:
        for ind in range(len(texts)):
            if len(texts[ind]) >= min_length:
                texts_long.append(texts[ind])
                labels_long.append(labels[ind])
    else:
        texts_long = texts
        labels_long = labels


    shuffle_ind = np.random.choice(len(texts_long), size = n_increase)
    for ind in shuffle_ind:
        text_copy = np.random.permutation(texts_long[ind])
        texts.append(text_copy)
        labels = np.append(labels, [labels_long[ind]])


    return texts, labels


def similar_augment(texts, labels, n_increase, n_word_replace, model_path, similar_threshold = 0.5):
    w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)

    shuffle_ind = np.random.choice(len(texts), size = n_increase)
    for ind in shuffle_ind:
        text_copy = copy.deepcopy(texts[ind])
        # if is_hier:

        replace_inds = np.random.choice(text_copy.shape[-1], size = n_word_replace, replace = False)
        for word_ind in replace_inds:
            word = text_copy[word_ind]
            try:
                closest, score = w2v.wv.most_similar(word)[0]
                if score > similar_threshold:
                    text_copy[word_ind] = closest
            except:
                continue

        texts.append(text_copy)
        labels = np.append(labels, [labels[ind]])

    return texts, labels


