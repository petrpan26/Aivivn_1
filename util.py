import pandas as pd


def refined(str):
    return str.replace('\"', '').replace('\n', '')


def read_file(file_name, is_train=True):
    with open(file_name) as fp:
        result_array = []
        while True:
            # Read name
            data_name = fp.readline()
            if (data_name == '\n'):
                continue
            if not data_name:
                break
            # Read text
            text = ""
            while text.count('\"') < 2:
                line = fp.readline()
                if (len(line) == 0):
                    break
                text += line
            if (len(text) == 0):
                break
            data = [refined(data_name), refined(text)]
            # Read label
            if is_train:
                label = fp.readline().replace('\"', '').replace('\n', '')
                data.append(label)
            result_array.append(data)
        columns = ['name', 'text', 'label'] if is_train else [
            'name', 'text'
        ]
        return pd.DataFrame(result_array, columns=columns)


print(read_file('train.crash', False)['text'][0])
