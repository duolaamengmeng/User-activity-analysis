import pandas as pd
import numpy as np
from datetime import datetime
import math


class AddNum:
    def __init__(self, filename, features, unique_ten_path):

        with open('/data/uspace/file/{}'.format(filename)) as data:
            self.df = pd.read_json(data, lines=True)
        self.features = features
        self.unique_ten = pd.read_csv(unique_ten_path)

    def pre_processing(self):
        self.df = self.df[self.features]
        ten_set = list(set(self.unique_ten['instance_id'].tolist()))
        instances = self.df['instance_id'].tolist()
        for index, i in enumerate(instances):
            if i not in ten_set:
                self.df = self.df.drop([index])

    def find_index(self, col_name):
        """ enters a column name in string format,
        returns the index of that feature in int"""
        return self.df.columns.tolist().index(col_name)

    def get_time(self):
        time_col = self.find_index('mtime')
        ts = self.df['mtime'].tolist()[0] / 1000
        ts = math.floor(ts)
        return datetime.fromtimestamp(ts).strftime('%Y%m%d')

    def add(self):
        self.pre_processing()
        users = list(set(self.df['user_id'].tolist()))
        all_users = self.df['user_id'].tolist()
        all_ten = self.df['instance_id'].tolist()
        array = np.array(self.df)
        data = []
        ten_col = self.find_index('instance_id')
        for index, i in enumerate(users):
            actions = 0
            all_index = []
            for indexJ, j in enumerate(all_users):
                if int(i) == int(j):
                    actions += 1
                    all_index.append(indexJ)
            data.append([i, actions, array[all_index[0], ten_col], self.get_time()])
        return data
