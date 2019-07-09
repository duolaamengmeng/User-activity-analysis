from __future__ import division

import time

import numpy as np
import pandas as pd
from add_feature import PreProcessing
from dataloader_1 import DataLoader
from set_year import SetYear


def process(file_path, num_of_work_day, tenant_path):
    d = np.load(file_path, allow_pickle=True)
    df = []
    for i in d:
        temp = list(i[0])
        temp.append(i[1])
        df.append(temp)
    print('start labelling dates...')
    df = pd.DataFrame(df, columns=['instance_id', 'user_id', 'date', 'actions'])
    pre_process = PreProcessing(df, 'date')
    df = pre_process.add_column()
    s = SetYear(df=df,
                filename=tenant_path)

    data_array = s.change_time()
    data_array = pd.DataFrame(data_array, columns=['instanceId', 'userId', 'date', 'actions'])
    data_array['date'] = pd.to_numeric(data_array['date'])
    data_array = data_array[(data_array['date'] >= 0) & (data_array['userId'] != 0) & (data_array['userId'] != '0')]

    # Get unique instances
    unique_instances = list(set(data_array['instanceId'].tolist()))
    print('cleaning...')

    def build(i):
        # select * from df where instanceId == i
        df = data_array[(data_array['instanceId'] == i)]

        d = df['date'].tolist()
        unique_dates = sorted(list(set(d)))

        # Make a dictionary containing {True_Date: index_of_date}
        dictionary = {}
        for index, item in enumerate(unique_dates):
            dictionary.update({item: index})
        column = []

        for i in d:
            for key in dictionary:
                if i == key:
                    column.append(dictionary[key])

        df['date'] = column
        return df

    result = list(map(build, unique_instances))

    for i in range(len(result)):
        if i == 0:
            data = result[i]
        else:
            data = pd.concat([data, result[i]], axis=0)

    data = data[data['date'] <= num_of_work_day]

    p = DataLoader(
        data, 'date', ['userId', 'actions', 'instanceId', 'date'],
        'instanceId'
    )
    data = p.sum_all()

    return data


if __name__ == '__main__':
    t = time.time()
    data = process('C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\data3.npy', 125,
                   'C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\instance_created.csv')
    np.save('data', data)
    print(time.time() - t)
