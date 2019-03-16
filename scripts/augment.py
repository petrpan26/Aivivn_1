import numpy as np
from gensim.models import KeyedVectors
import copy
import random
from gensim.similarities.index import AnnoyIndexer


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


def similar_augment(texts, labels, n_increase, n_word_replace, model_path, similar_threshold = 0.5, use_annoy = True, annoy_path = None):
    w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)
    texts_long = []
    labels_long = []
    if use_annoy:
        if annoy_path is None:
            indexer = AnnoyIndexer(w2v, 100)
        else:
            indexer = AnnoyIndexer()
            indexer.load(annoy_path)

    for ind in range(len(texts)):
        if len(texts[ind]) >= n_word_replace:
            texts_long.append(texts[ind])
            labels_long.append(labels[ind])

    shuffle_ind = np.random.choice(len(texts_long), size = n_increase)
    for ind in shuffle_ind:
        text_copy = copy.deepcopy(texts_long[ind])
        # if is_hier:

        replace_inds = np.random.choice(text_copy.shape[-1], size = n_word_replace, replace = False)
        for word_ind in replace_inds:
            word = text_copy[word_ind]
            try:

                closest, score = w2v.wv.most_similar(
                    word, topn = 2,
                    indexer = indexer if use_annoy else None
                )[1]
                if score > similar_threshold:
                    text_copy[word_ind] = closest
            except:
                continue

        texts.append(text_copy)
        labels = np.append(labels, [labels_long[ind]])

    return texts, labels




def create_sim_dict(word_map, model_path, similar_threshold = 0.5, use_annoy = True, annoy_path = None):
    w2v = KeyedVectors.load_word2vec_format(model_path, binary=True)
    if use_annoy:
        if annoy_path is None:
            indexer = AnnoyIndexer(w2v, 100)
        else:
            indexer = AnnoyIndexer()
            indexer.load(annoy_path)

    sim_dict = dict()
    for word in word_map:
        try:
            closest, score = w2v.wv.most_similar(
                word, topn=2,
                indexer=indexer if use_annoy else None
            )[1]
            if score > similar_threshold and closest in word_map:
                sim_dict[word_map[word]] = word_map[closest]
        except:
            continue

    return sim_dict

def similar_augment_from_sim_dict(texts, labels, sim_dict, n_increase, keep_prob = 0.5):
    aug_ind = np.random.choice(len(texts), size = n_increase)
    i = -1
    for ind in aug_ind:
        i += 1
        text_aug = copy.deepcopy(texts[ind])
        for word_ind in range(len(text_aug)):
            word = text_aug[word_ind]
            if word in sim_dict:
                p = random.uniform(0, 1)
                if p > keep_prob:
                    text_aug[word_ind] = sim_dict[word]

        texts = np.append(texts, [text_aug], axis = 0)
        labels = np.append(labels, [labels[ind]], axis = 0)

    return texts, labels






