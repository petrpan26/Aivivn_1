import pandas as pd
import copy
import os
from pathlib2 import Path


def refined(str):
    return str.replace('\"', '').replace('\n', '')


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
    with open(file_name) as fp:
        data_lines = filter(lambda x: x != '', fp.read().split('\n'))
        pattern = 'train' if is_train else 'test'
        datas = split_array(data_lines, lambda x: pattern in x)
        if is_train:
            result_array = map(
                lambda x: [x[0], ' '.join(x[1:-1]), int(x[-1])], datas)
        else:
            result_array = map(
                lambda x: [x[0], ' '.join(x[1:-1])], datas
            )
        columns = ['name', 'text', 'label'] if is_train else [
            'name', 'text'
        ]
        return pd.DataFrame(result_array, columns=columns)


path = os.getcwd()
if path.endswith('scripts'):
    path = os.path.abspath('..')
train_path = os.path.join(path, 'data/test.crash')
print(read_file(train_path, False)
      ['name'].describe())
