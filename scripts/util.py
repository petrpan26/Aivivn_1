import pandas as pd
import copy
import os
import numpy as np

from spacy.lang.vi import Vietnamese
from spacy.attrs import ORTH, LEMMA
from .constant import EMOTICONS



def split_array(arr, condition):
    if len(arr) == 0:
        return []
    result = []
    accumulated = [arr[0]]
    for ele in arr[1:]:
        if condition(ele):
            result.append(copy.deepcopy(accumulated))
            accumulated = [copy.deepcopy(ele)]
        else:
            accumulated.append(copy.deepcopy(ele))
    result.append(copy.deepcopy(accumulated))
    return result


def read_file(file_name, is_train=True):
    data_lines = list(
        filter(lambda x: x != '', open(file_name).read().split('\n')))
    pattern = 'train' if is_train else 'test'
    datas = split_array(data_lines, lambda x: pattern in x)
    if is_train:
        result_array = map(
            lambda x: [x[0], ' '.join(x[1:-1]), int(x[-1])], datas)
    else:
        result_array = map(lambda x: [x[0], ' '.join(x[1:-1])], datas)
    columns = ['name', 'text', 'label'] if is_train else ['name', 'text']
    return pd.DataFrame(result_array, columns=columns)


def tokenize(texts):
    ExceptionsSet = {}
    for orth in EMOTICONS:
        ExceptionsSet[orth] = [{ORTH: orth}]


    nlp = Vietnamese()
    tokenizer = nlp.create_pipe("tokenizer")
    for emoticon in EMOTICONS:
        tokenizer.add_special_case(emoticon, ExceptionsSet[emoticon])
    docs = []
    for text in texts:
        tokens = np.array([token.text for token in tokenizer(text)[1:-1]])
        docs.append(tokens)

    return np.array(docs)
