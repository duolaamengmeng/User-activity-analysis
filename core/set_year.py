from __future__ import division
import numpy as np
import pandas as pd
from datetime import date
import time
import math


class SetYear:
    def __init__(self, df, filename):
        # only include work days
        self.df = df[(df['date_type'] == 0) | (df['date_type'] == 2)].iloc[:, :4]
        self.time_table = np.array(pd.read_csv(filename))

    def makedict(self):
        dictionary = {}
        for i in range(len(self.time_table)):
            # Dictionary { instanceId: created_time (in unix timestamp)}
            dictionary.update({self.time_table[i, 0]: self.time_table[i, 1]})
        return dictionary

    def unix_time(self):
        df = self.df
        time_col = list(set(df['date'].tolist()))
        t = {}
        for i in time_col:
            i = str(i)
            d = date(int(i[0:4]), int(i[4:6]), int(i[6:8]))
            t.update({i: time.mktime(d.timetuple())})

        all_time = []
        for i in df['date'].tolist():
            for key in t:
                if str(i) == key:
                    all_time.append(t[key])

        df['date'] = all_time

        return df

    def change_time(self):
        df = self.unix_time()
        dictionary = self.makedict()
        array = np.array(df)
        print('start step 2')
        t = time.time()
        for index, i in enumerate(array):
            if index % 10000 == 9999:
                print(index, time.time() - t, len(array))
                t = time.time()

            for key in dictionary:
                if i[2] == key:
                    array[index, 3] = math.ceil(int(
                        array[index, 3]) - int(dictionary[key]) / 88400)
        return array
