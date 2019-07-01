from __future__ import division
import time
import numpy as np
import pandas as pd
from IPython.display import clear_output
from librosa.feature import mfcc


class DataLoader:
    def __init__(self, filename, time_col, features, ten_col,
                 shuffle=True, batch_size=64):
        """A class for pre-processing, generating data for time series analysis


        filename-str: path to the csv format data file
        new_buyers-list of str- contains tenant ID of newly purchased tenants
        time_col-str: column name of time steps
        onehot_features-list of str: features that needs to be one hot encoded
        features-list of str: list of features to be selected
        ten_col-str: column name that contains tenant ID
        error-list of str: error codes to be converted
        error_col-str: column name of which contains error codes
        data_split-bool: whether the data needs to be split, default True
        sample_size-int: sample size of the split data, default 500
        shuffle-bool: whether needs to shuffle in the batch generator, default True
        batch_size: batch size of each output of the batch generator, default 64"""

        self.df = filename
        self.time_col = time_col
        self.features = features
        self.ten_col = ten_col
        self.shuffle = shuffle
        self.batch_size = batch_size

    def find_time_bin(self):
        """returns a list containing time-steps"""
        return set(self.df[self.time_col])

    def find_index(self, col_name):
        """ enters a column name in string format,
        returns the index of that feature in int"""
        return self.df.columns.tolist().index(col_name)

    def pre_processing(self):
        """takes a data frame,
        returns a one hot encoded Sci-py sparse matrix"""
        # get features
        self.df = self.df[self.features]
        data = np.array(self.df)

        return data

    def time(self):
        array = self.pre_processing()
        time_col = self.find_index(self.time_col)
        time_bin = list(self.find_time_bin())
        total_iteration = len(time_bin)
        result = []
        # iterate through each timestep, create a new dimension for time step
        for index, i in enumerate(time_bin):
            if index == 0:
                t = time.time()
            clear_output(wait=True)
            print('step 1: iteration {}  out of {}'.format(index, total_iteration))
            print('time ultilized: {}'.format(time.time() - t))
            t = time.time()
            buff = []
            for j, item in enumerate(array):
                if item[time_col] == i:  # get only value, preventing from making another matrix
                    buff.append(item)

            result.append(buff)

        return result

    def make3dts(self):
        """This method reshape the dataset from (TimeStep,-1.FeatureSpace)
        to (TimeStep,Company,-1,FeatureSpace)"""
        lst = self.time()
        ten = self.find_index(self.ten_col)
        unique_ten = list(set(self.df.iloc[:, ten]))
        total_iteration = len(lst)
        for j_index, j in enumerate(lst):  # iterate time step
            if j_index == 0:
                t = time.time()
            clear_output(wait=True)
            print('step 2: iteration {}  out of {}'.format(j_index, total_iteration))
            print('time ultilized: {}'.format(time.time() - t))
            t = time.time()

            cpp = []
            for i in unique_ten:  # iterate unique companies
                cp = []
                for k_index, k in enumerate(j):  # iterate each element in time step
                    if int(k[ten]) == int(i):
                        cp.append(lst[j_index][k_index])

                cpp.append(cp)
            lst[j_index] = cpp
        return lst

    def sum_all(self):
        """This method reshapes the dataset to (TimeStep, Company, FeatureSpace)
        by combines all information of each (TimeStep, Company) into a single entry.
        As for now, the combination is a matrix dot product of (vector 'num_of_operation' and
        feature matrix) of each (TimeStep, Company)"""

        data = self.make3dts()
        d = []
        for indx, i in enumerate(data):
            buff = []
            for j, item in enumerate(i):
                multiplier = []

                for k in item:
                    multiplier.append(int(k[1]))

                buff.append(sum(multiplier))
            d.append(buff)
        return d

    def frequency_feature(self):
        data = np.array(self.sum_all())
        data = np.transpose(data, (1, 0))
        data_mfccs = []
        for i in data:
            sig = i / max(abs(i))
            data_mfccs.append(mfcc(sig, sr=10, n_mfcc=2, hop_length=10))
        data_mfccs = np.array(data_mfccs)
        # data_mfccs = np.transpose(data_mfccs,(0,2,1))
        return data_mfccs
