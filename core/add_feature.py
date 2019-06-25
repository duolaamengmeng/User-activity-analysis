import json
from datetime import datetime
from urllib.request import urlopen, Request
import pandas as pd
from IPython.display import clear_output


class PreProcessing:
    def __init__(self, filename, time_col):
        self.df = pd.read_table(filename)
        self.time_col = time_col
        self.time_set = list(set(self.df[self.time_col].tolist()))

    def pre_processing(self):

        date_time = []
        for index, i in enumerate(self.time_set):
            ts = int(i)
            date_time.append(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))
        return date_time

    def find_date_type(self):
        server_url = "http://api.goseek.cn/Tools/holiday?date="
        date_time = self.pre_processing()

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
        for index, i in enumerate(self.df[self.time_col].tolist()):

            clear_output(wait=True)
            print('step 1: iteration {}  out of {}'.format(index, len(self.df[self.time_col].tolist())))

            for j in keys:
                if i == j:
                    date_type.append(dictionary.get(j))

        df = pd.concat([self.df, pd.DataFrame(date_type, columns='data_type')], axis=1)
        return df
