import json
from datetime import datetime
from urllib.request import urlopen, Request

import pandas as pd
from IPython.display import clear_output


class PreProcessing:
    def __init__(self, filename, time_col, unix_time_stamp=False):
        """
        this class takes a DataFrame and its time column name as input,
        requests the date type on the server "http://api.goseek.cn/Tools/holiday?date="
        , makes a new column for the DataFrame which describes date type.

        :param filename: a DataFrame
        :param time_col: the name of the column that records time stamp in str
        :param unix_time_stamp: whether the format of time_col is in unix time stamp, bool

        正常工作日对应结果为 0,
        法定节假日对应结果为 1,
        节假日调休补班对应的结果为 2，
        休息日对应结果为 3

        """
        self.df = filename
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
        """
        this method request date type from server, makes a dictionary
        that has every unique date and its date type pair

        :return: a dictionary {unique_dates:date_type}
        """

        server_url = "http://api.goseek.cn/Tools/holiday?date="

        # if time format is already '%Y%m%d', no need to convert
        if self.unix_time_stamp:
            date_time = self.pre_processing()
        else:
            date_time = list(self.time_set)

        # create an empty dictionary
        dictionary = {}
        for index, date in enumerate(date_time):
            vop_url_request = Request(server_url + date)

            # allow maximum of 10 tries
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
            # append keys and value pairs to dictionary
            # {date,date_type}
            dictionary.update({self.time_set[index]: json.loads(vop_response.read())['data']})
            clear_output(wait=True)
            print('step 1: iteration {}  out of {}'.format(index, len(date_time)))

        return dictionary

    def add_column(self):
        """
        This method uses a dictionary as input (output of self.find_date_type),
        adds feature date_type to the original DataFrame.

        :return: a DataFrame with date_type column
        """
        dictionary = self.find_date_type()

        # Make an empty list to store date type
        date_type = []
        keys = list(dictionary.keys())
        total_iteration = len(self.df[self.time_col].tolist())

        # iterate through time column
        for index, i in enumerate(self.df[self.time_col].tolist()):
            # Print progress every 1000 iterations
            if index % 1000 == 999:
                clear_output(wait=True)
                print('step 1: iteration {}  out of {}'.format(index, total_iteration))

            # iterate through keys (unique dates) in the dictionary
            # if the date in time_col == date in dictionary, append date type to the empty list
            for j in keys:
                if i == j:
                    date_type.append(dictionary[j])
        # Finally concatenate the empty list with the original DataFrame on axis 1
        df = pd.concat([self.df, pd.DataFrame(date_type, columns=['date_type'])], axis=1)
        return df
