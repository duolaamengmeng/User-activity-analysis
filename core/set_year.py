from __future__ import division
import numpy as np
import pandas as pd
from datetime import date
import time
import math


class SetYear:
    def __init__(self, df, filename):
        # only include work days
        self.df = df[(df['date_type'] == 0) | (df['date_type'] == 2)].iloc[:, :5]
        self.time_table = np.array(pd.read_csv(filename), dtype=str)

    def makedict(self):
        dictionary = {}
        for i in range(len(self.time_table)):
            # Generate a dictionary that stores the start time of each customer in unix time stamp
            # Dictionary { instanceId: created_time (in unix timestamp)}
            dictionary.update({self.time_table[i, 0]:
                                   int(time.mktime(time.strptime(self.time_table[i, 1], '%Y/%m/%d %H:%M')))})
        return dictionary

    def find_index(self, col_name):
        """ enters a column name in string format,
        returns the index of that feature in int"""
        return self.df.columns.tolist().index(col_name)

    def unix_time(self):
        df = self.df
        # get
        time_col = list(set(df['date'].tolist()))
        t = {}
        for i in time_col:
            i = str(i)
            d = date(int(i[0:4]), int(i[4:6]), int(i[6:8]))
            t.update({i: time.mktime(d.timetuple())})

        all_time = []

        for i in df['date'].tolist():
            all_time.append(t[str(i)])

        df['date'] = all_time

        return df

    def change_time(self):
        print('change to unix time...')
        df = self.unix_time()
        print('making dictionary...')
        dictionary = self.makedict()
        print('labelling...')
        array = np.array(df)

        date_col = self.find_index('date')
        ten_col = self.find_index('instance_id')

        def build(i):
            return math.ceil((int(i[date_col]) - int(dictionary[str(i[ten_col])])) / 86400)
        result = list(map(build, array))
        array[:, date_col] = result

        return array
