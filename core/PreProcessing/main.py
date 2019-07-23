from make_num import AddNum
import os
import numpy as np
import pandas as pd
import time

def loop(file_path, feature, unique_ten_path, apptype_path):
    file_names = sorted(os.listdir(file_path))
    broken_file = []
    for index, i in enumerate(file_names):
        d = AddNum(file_path, filename=i, features=feature, unique_ten_path=unique_ten_path, apptype_path=apptype_path)
        os.system('cls' if os.name == 'nt' else 'clear')
        print('file No.{} out of {}'.format(index, len(file_names)))
        temp = d.add()
        if index == 0:
            # initialization
            data = temp
        else:
            try:
                data = np.concatenate([data, temp], axis=0)

            except:
                data = data

    return data


if __name__ == '__main__':
    t = time.time()
    data = loop(file_path='/data/uspace/file', feature=['user_id', 'instance_id', 'mtime', 'open_appid'],
                unique_ten_path='/data/uspace/code/uspace.csv', apptype_path='/data/uspace/code/appType.csv')
    np.save('data.npy', data)
    print(time.time() - t)
