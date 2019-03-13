import numpy as np

def shuffle_augment(texts, labels, n_increase, only_long = True):
    texts_long = []
    labels_long = []

    if only_long:
        for ind in range(len(texts)):
            if len(texts[ind]) > 1:
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
