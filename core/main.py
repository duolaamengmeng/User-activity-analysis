from make_num import AddNum
import os
import pandas as pd
import numpy as np


def loop(file_path, feature, unique_ten_path):
    file_names = os.listdir(file_path)
    print(file_names)
    for index, i in enumerate(file_names):
        print('File No.{}'.format(index))
        d = AddNum(filename=i, features=feature, unique_ten_path=unique_ten_path)
        da = d.add()
        if index == 0:
            data = np.concatenate([np.zeros(0, 4), da], axis=0)
        else:
            data = np.concatenate([data, d], axis=0)
    return data


if __name__ == '__main__':
    data = loop(file_path='/data/uspace/file', feature=['user_id', 'instance_id', 'mtime'],
                unique_ten_path='/data/uspace/code')
    np.save('data.npy', data)
