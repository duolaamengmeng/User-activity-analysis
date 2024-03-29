import pandas as pd
import numpy as np
from datetime import datetime
import math
from progress.bar import Bar


class AddNum:
    def __init__(self, file_path, filename, features, unique_ten_path, apptype_path):
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
        self.appType = pd.read_csv(apptype_path)

    def pre_processing(self):
        print('total number of line: {} '.format(self.df.shape[0]))
        df = self.df
        df = df[self.features]
        original_len = df.shape[0]
        df = df.dropna()
        new_len = df.shape[0]
        df['instance_id'] = df['instance_id'].astype(str)
        self.unique_ten['instance_id'] = self.unique_ten['instance_id'].astype(str)
        df = pd.merge(self.unique_ten, df, how='inner', on='instance_id')
        print('Num of selected line: {} '.format(df.shape[0]))

        print('Num of NA deleted: {} '.format(original_len - new_len))

        df['open_appid'] = pd.to_numeric(df['open_appid']).dropna()
        dict = {}
        instance_table = np.array(self.appType, dtype=float)
        for i in range(instance_table.shape[0]):
            # update dictionary {OpenAppID: APPType}
            dict.update({instance_table[i, 0]: instance_table[i, 1]})

        app = []

        appId = df['open_appid'].tolist()
        for i in appId:
            if i in dict.keys():
                app.append(dict[i])
            else:
                app.append(i)

        df['open_appid'] = app

        df['date'] = self.get_time()
        self.df = df

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

        # data table
        array = np.array(self.df)
        # column index that stores instanceId
        ten_col, user_col, date_col, app_id = self.find_index('instance_id'), self.find_index(
            'user_id'), self.find_index(
            'date'), self.find_index('open_appid')
        # date, same across all instances

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

        data = []
        for key in dict:
            data.append([key, dict[key]])

        return data

