from __future__ import division

import time

import numpy as np
import pandas as pd
from add_feature import PreProcessing
from dataloader_1 import DataLoader
from set_year import SetYear


def process(file_path, num_of_work_day, tenant_path):
    """

    :param file_path: path of the PreProcessed npy file
    :param num_of_work_day: number of workdays to be included in the final output
    :param tenant_path: path of the file which contains [instanceId, activation]
    :return: data: a nested list;
    """

    # Raw Data contains all individual operations, saved in (# of days) files
    # For each file(one day of raw data), information of each user is collected
    # Previous step is ran (# of days) time, result is concatenated on the 0th axis
    # Preprocessed data contains information of each user on each day, with instanceId intact

    # Load data preprocessed from server
    d = np.load(file_path, allow_pickle=True)
    df = []
    for i in d:
        temp = list(i[0])
        temp.append(i[1])
        df.append(temp)

    print('start labelling dates...')

    # Transform into DataFrame
    df = pd.DataFrame(df, columns=['instance_id', 'user_id', 'date', 'appid', 'actions'])
    df = df.fillna('892714')
    # Add a column to df which describes date type
    # Request from URL, make sure internet is stable
    pre_process = PreProcessing(df, 'date')
    # 正常工作日对应结果为 0,
    # 法定节假日对应结果为 1,
    # 节假日调休补班对应的结果为 2，
    # 休息日对应结果为 3
    df = pre_process.add_column()

    # select only workdays
    # for every instance, make activation date day 0
    s = SetYear(df=df,
                filename=tenant_path)
    data_array = s.change_time()
    data_array = pd.DataFrame(data_array, columns=['instanceId', 'userId', 'date', 'appid', 'actions'])
    data_array['date'] = pd.to_numeric(data_array['date'])

    # Only select dates after activation date
    # Drop userId=0
    data_array = data_array[(data_array['date'] >= 0) & (data_array['userId'] != 0) & (data_array['userId'] != '0')]

    # Get unique instances
    unique_instances = list(set(data_array['instanceId'].tolist()))
    print('cleaning...')

    # value of the date is inaccurate with non-workdays excluded since it is calculated as:
    # date = currentTS - activationDateTS
    # Hence, change the value of the dates to index of the date
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
            # for key in dictionary:
            #     if i == key:
            column.append(dictionary[i])

        df.loc[:, 'date'] = column
        return df
    result = list(map(build, unique_instances))
    print('sorting...')
    # Concatenate results from map on the 0 axis
    for i in range(len(result)):
        if i == 0:
            data = result[i]
        else:
            data = pd.concat([data, result[i]], axis=0)

    # Select desirable amount of workdays
    data = data[data['date'] <= num_of_work_day]

    # Pass data to data loader, returns: data(2d nested list) & unique_instance (list)
    # data has shape of (time_step, instance)
    p = DataLoader(
        data, 'date', ['userId', 'actions', 'instanceId', 'date', 'appid'],
        'instanceId', onehot_features=['appid']
    )
    data, unique_ten = p.sum_all()

    return data, unique_ten


if __name__ == '__main__':
    t = time.time()
    data, unique_ten = process('C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\data_all.npy', 125,
                   'C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\instance_created.csv')
    np.save('data', data)
    np.save('instance', unique_ten)
    print(time.time() - t)
