import json
from datetime import datetime
from urllib.request import urlopen, Request
import pandas as pd
from IPython.display import clear_output


class PreProcessing:
    def __init__(self, filename, time_col, unix_time_stamp=False):
        self.df = pd.read_table(filename)
        self.time_col = time_col
        self.time_set = list(set(self.df[self.time_col].tolist()))
        self.unix_time_stamp = unix_time_stamp

    def pre_processing(self):
        date_time = []
        for index, i in enumerate(self.time_set):
            ts = int(i)
            date_time.append(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))
        return date_time

    def find_date_type(self):
        server_url = "http://api.goseek.cn/Tools/holiday?date="
        if self.unix_time_stamp:
            date_time = self.pre_processing()
        else:
            date_time = list(self.time_set)
        dictionary = {}

        for index, date in enumerate(date_time):
            vop_url_request = Request(server_url + date)
            tries = 10
            for i in range(tries):
                try:
                    vop_response = urlopen(vop_url_request)
                except KeyError as e:
                    if i < tries - 1:  # i is zero indexed
                        continue
                    else:
                        raise
                break

            dictionary.update({self.time_set[index]: json.loads(vop_response.read())['data']})
            clear_output(wait=True)
            print('step 1: iteration {}  out of {}'.format(index, len(date_time)))

        return dictionary

    def add_column(self, dictionary):
        date_type = []
        keys = list(dictionary.keys())
        total_iteration = len(self.df[self.time_col].tolist())
        for index, i in enumerate(self.df[self.time_col].tolist()):

            if index % 1000 == 999:
                clear_output(wait=True)
                print('step 1: iteration {}  out of {}'.format(index, total_iteration))

            for j in keys:
                if i == j:
                    date_type.append(dictionary[j])

        df = pd.concat([self.df, pd.DataFrame(date_type, columns=['date_type'])], axis=1)
        return df
