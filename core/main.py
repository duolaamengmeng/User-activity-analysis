from __future__ import division

import time

import numpy as np
import pandas as pd
from add_feature import PreProcessing
from dataloader_1 import DataLoader
from set_year import SetYear


def dataloader(apptype_path, file_path, num_of_work_day, tenant_path):
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
    # In order to reduce dimension of feature space, apps are grouped into 3 categories:
    # {0: 一般应用， 1: 重要应用， 3: 核心应用}

    # Load and make appType lookup table
    appType = pd.read_csv(apptype_path)
    instance_table = np.array(appType, dtype=float)
    dict = {}
    for i in range(instance_table.shape[0]):
        # update dictionary {OpenAppID: APPType}
        dict.update({instance_table[i, 0]: instance_table[i, 1]})

    # Transform into DataFrame
    df = pd.DataFrame(df, columns=['instance_id', 'user_id', 'date', 'appid', 'actions'])
    # mark NA with my magic number
    df = df.fillna('892714')
    # Extract Appid Column to a list
    app = []
    appId = df['appid'].tolist()
    # substitute all appid with appType
    for i in appId:
        if i in dict.keys():
            app.append(dict[i])
        else:
            app.append(i)
    df['appid'] = app
    #  eliminate all appID that is not present in the lookup table (approximately 0.01%)
    df = pd.merge(df, pd.DataFrame([0, 1, 2], columns=['appid'], dtype=object), how='inner', on='appid')

    print('start labelling dates...')
    # Add a column to df which describes date type
    # Request from URL, make sure internet is stable
    pre_process = PreProcessing(df, 'date', request_from_server=False)
    # 正常工作日对应结果为 0,
    # 法定节假日对应结果为 1,
    # 节假日调休补班对应的结果为 2，
    # 休息日对应结果为 3
    df = pre_process.add_column()
    # select only workdays
    # for every instance, make activation date day 0
    s = SetYear(df=df, filename=tenant_path)
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

    ##################################################
    # This part needs to be rewritten, extremely slow#
    ##################################################
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

    # # Select desirable amount of workdays
    data = data[data['date'] <= num_of_work_day]

    # Pass data to data loader, returns: data(2d nested list) & unique_instance (list)
    # Data has shape of (time_step, instance)

    ###################################################
    # This class needs to be rewritten, extremely slow#
    ###################################################
    p = DataLoader(
        data, 'date', ['userId', 'actions', 'instanceId', 'date', 'appid'],
        'instanceId', onehot_features=['appid']
    )
    data, unique_ten = p.sum_all()

    return data, unique_ten


if __name__ == '__main__':
    t = time.time()
    data, unique_ten = dataloader('C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\appType.csv',
                               'C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\data_all.npy', 125,
                               'C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\instance_created.csv')
    np.save('data', data)
    np.save('instance', unique_ten)

    print(time.time() - t)
