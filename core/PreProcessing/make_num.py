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

        ten_set = np.array(list(set(self.unique_ten['instance_id'].tolist())), dtype=int)
        instances = self.df['instance_id'].tolist()
        data = []
        bar = Bar('Pre-process', max=len(instances))
        for index, i in enumerate(instances):
            if int(i) in ten_set:
                data.append(array[index, :])
            bar.next()
        bar.finish()
        self.df = pd.DataFrame(data, columns=self.features)

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
        users = list(set(self.df['user_id'].tolist()))
        all_users = self.df['user_id'].tolist()
        all_ten = self.df['instance_id'].tolist()
        array = np.array(self.df)
        data = []
        ten_col = self.find_index('instance_id')

        bar = Bar('Add Column', max=len(users))
        for index, i in enumerate(users):
            actions = 0
            all_index = []
            for indexJ, j in enumerate(all_users):
                if i == j:
                    actions += 1
                    all_index.append(indexJ)
            data.append([i, actions, array[all_index[0], ten_col], self.get_time()])
            bar.next()
        bar.finish()
        return data
