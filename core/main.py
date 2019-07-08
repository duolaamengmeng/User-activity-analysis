from __future__ import division
import numpy as np
import pandas as pd
from add_feature import PreProcessing
from dataloader_1 import DataLoader
from set_year import SetYear
import matplotlib.pyplot as plt


def process(file_path, num_of_work_day, tenant_path):
    data = np.load(file_path)
    df = pd.DataFrame(data, columns=['userId', 'actions', 'instanceId', 'date'])
    pre_process = PreProcessing(df, 'date')
    df = pre_process.add_column()
    s = SetYear(df=df,
                filename=tenant_path)

    data_array = s.change_time()
    data_array = pd.DataFrame(data_array, columns=['userId', 'actions', 'instanceId', 'date'])
    data_array['date'] = pd.to_numeric(data_array['date'])
    data_array = data_array[(data_array['date'] >= 0) & (data_array['userId'] != '0')]

    # Get unique instances
    unique_instances = list(set(data_array['instanceId'].tolist()))

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
    return data


if __name__ == '__main__':
    data = process('C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\data_1.npy', 125,
                   'C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\instance_created.csv')
    np.save('data', data)
