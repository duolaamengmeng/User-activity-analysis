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
        self.time_table = np.array(pd.read_csv(filename), dtype=str)

    def makedict(self):
        dictionary = {}
        for i in range(len(self.time_table)):
            # Generate a dictionary that stores the start time of each customer in unix time stamp
            # Dictionary { instanceId: created_time (in unix timestamp)}
            dictionary.update({self.time_table[i, 0]:
                                   int(time.mktime(time.strptime(self.time_table[i, 1], '%Y/%m/%d %H:%M')))})
        return dictionary

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
            for key in t:
                if str(i) == key:
                    all_time.append(t[key])

        df['date'] = all_time

        return df

    # def change_time(self):
    #     df = self.unix_time()
    #     dictionary = self.makedict()
    #     array = np.array(df)
    #     print('start step 2')
    #     t = time.time()
    #
    #     # iterate through data set
    #     for index, i in enumerate(array):

    #         # iterate through keys
    #         for key in dictionary:
    #             # if company id matches
    #             if str(i[2]) == str(key):
    #                 # operations
    #                 array[index, 3] = math.ceil((int(
    #                     array[index, 3]) - int(dictionary[key])) / 86400)
    #
    #     return array

    def change_time(self):
        df = self.unix_time()
        dictionary = self.makedict()
        array = np.array(df)
        print('start step 2')
        t = time.time()

        def build(i):
            for key in dictionary:
                if str(i[2]) == str(key):
                    return math.ceil((int(
                        i[3]) - int(dictionary[key])) / 86400)

        result = list(map(build, array))
        array[:, 3] = result

        return array
