from make_num import AddNum
import os
import numpy as np
from multiprocessing import Process, Queue


def loop(file_path, feature, unique_ten_path):
    file_names = os.listdir(file_path)
    print(file_names)
    for index, i in enumerate(file_names):
        print('file No.{} out of {}'.format(index, len(file_names)))
        d = AddNum(file_path, filename=i, features=feature, unique_ten_path=unique_ten_path)
        da = d.add()
        if index == 0:
            data = da
        else:
            data = np.concatenate([data, da], axis=0)
        np.save('data.npy', data)
    return data


if __name__ == '__main__':
    data = loop(file_path='/data/uspace/file', feature=['user_id', 'instance_id', 'mtime'],
                unique_ten_path='/data/uspace/code/uspace.csv')
    np.save('data.npy', data)
