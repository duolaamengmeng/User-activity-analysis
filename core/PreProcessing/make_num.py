import pandas as pd
import numpy as np
from datetime import datetime
import math
from progress.bar import Bar


class AddNum:
    def __init__(self, file_path, filename, features, unique_ten_path):
        """
        This class takes a Json file, transform the instances from individual
        operations to users, column 'action' which describes the number of actions
        of a user in a day is generated.

        :param file_path: path where the files are being saved
        :param filename: name of the individual file in Json format
        :param features: features to be selected, in a list str format
        :param unique_ten_path: path of the file that contains tenants which will be selected

        """

        with open('{}/{}'.format(file_path, filename)) as data:
            self.df = pd.read_json(data, lines=True)
        self.features = features
        self.unique_ten = pd.read_csv(unique_ten_path)

    def pre_processing(self):
        self.df = self.df[self.features]
        array = np.array(self.df)
        self.df['instance_id'] = pd.to_numeric(self.df['instance_id'])
        self.unique_ten['instance_id'] = pd.to_numeric(self.unique_ten['instance_id'])
        instances = self.df['instance_id'].tolist()
        ten_set = set(self.unique_ten['instance_id'].tolist())
        data = []
        # bar = Bar('Pre-process', max=len(instances))
        # for index, i in enumerate(instances):
        #     if int(i) in ten_set:
        #         data.append(array[index, :])
        #     bar.next()
        # bar.finish()
        # self.df = pd.DataFrame(data, columns=self.features)
        self.df = pd.merge(self.unique_ten, self.df, how='inner', on='instance_id')
        self.df['date'] = self.get_time()

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
        print('pre-process finished')
        # unique users
        users = set(self.df['user_id'].tolist())
        # all users
        all_users = self.df['user_id'].tolist()
        # data table
        array = np.array(self.df)
        data = []
        # column index that stores instanceId
        ten_col, user_col, date_col, app_id = self.find_index('instance_id'), self.find_index(
            'user_id'), self.find_index(
            'date'), self.find_index('open_appid')
        # date, same across all instances
        date = self.get_time()
        # select desirable columns
        iterator = np.array([array[:, i] for i in [ten_col, user_col, date_col, app_id]]).T
        dict = {}
        bar = Bar('Add Column', max=len(list(iterator)))
        for i in iterator:
            key = (i[0], i[1], i[2], i[3])
            if key in dict.keys():
                dict[key] += 1
            else:
                dict[key] = 1
            bar.next()
        bar.finish()
        # # iterate unique user
        # for index, i in enumerate(users):
        #     actions = 0
        #     all_index = []
        #     # iterate through the whole data set
        #     for indexJ, j in enumerate(all_users):
        #         if i == j:
        #             actions += 1
        #             all_index.append(indexJ)

        # [userId, #of actions, instanceId, date]
        # data.append([i, actions, array[all_index[0], ten_col], date])
        # bar.next()
        # bar.finish()
        # data = [[key, dictionary[key]] for key in dictionary]
        data = []
        for key in dict:
            data.append([key, dict[key]])

        return data

