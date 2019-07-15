from __future__ import division
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from IPython.display import clear_output
from librosa.feature import mfcc
from scipy.sparse import hstack


class DataLoader:
    def __init__(self, filename, time_col, features, ten_col, onehot_features,
                 shuffle=True, batch_size=64):
        """

        :param filename: A DataFrame to be passed in
        :param time_col: Name of the column that describes time steps
        :param features: Features that we are interested in
        :param ten_col: Name of the column that describes company(instance) identification
        :param onehot_features: List of the categorical variables' name (str) that needs to be one hot encoded
        :param shuffle: whether shuffle for batch generator
        :param batch_size: batch size of the batch generator
        """

        self.df = filename
        self.time_col = time_col
        self.features = features
        self.ten_col = ten_col
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.onehot_features = onehot_features

    def find_time_bin(self):
        """returns a list containing time-steps"""
        return sorted(list(set(self.df[self.time_col])), key=int)

    def find_index(self, col_name):
        """ enters a column name in string format,
        returns the index of that feature in int"""
        return self.df.columns.tolist().index(col_name)

    def pre_processing(self):
        """
        select desired features
        :return: a numpy array
        """
        # get features
        self.df = self.df[self.features]
        print(self.df.head())
        for i in self.onehot_features:
            self.df[i] = self.df[i].astype(str)
            one_hot = pd.get_dummies(self.df[i])
            self.df = self.df.drop(i, axis=1)
            self.df = self.df.join(one_hot)
        print(self.df.head())


    def time(self):
        self.pre_processing()
        array = np.array(self.df)
        time_col = self.find_index(self.time_col)
        time_bin = self.find_time_bin()
        total_iteration = len(time_bin)
        result = []
        # iterate through each timestep, create a new dimension for time step
        for index, i in enumerate(time_bin):
            if index == 0:
                t = time.time()
            clear_output(wait=True)
            print('step 1: iteration {}  out of {}'.format(index, total_iteration))
            print('time utilized: {}'.format(time.time() - t))
            t = time.time()
            buff = []
            for j, item in enumerate(array):
                if item[time_col] == i:
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
            print('time utilized: {}'.format(time.time() - t))
            t = time.time()

            cpp = []
            for i in unique_ten:  # iterate unique companies
                cp = []
                for k_index, k in enumerate(j):  # iterate each element in time step
                    if int(k[ten]) == int(i):
                        cp.append(lst[j_index][k_index])

                cpp.append(cp)
            lst[j_index] = cpp
        return lst, unique_ten

    def sum_all(self):
        """This method reshapes the dataset to (TimeStep, Company, FeatureSpace)
        by combines all information of each (TimeStep, Company) into a single entry.
        As for now, the combination is a matrix dot product of (vector 'num_of_operation' and
        feature matrix) of each (TimeStep, Company)"""

        data, unique_ten = self.make3dts()
        action_col = self.find_index('actions')
        d = []

        onehot_col = [self.find_index(z) for z in self.df.columns if z not in self.features]
        for indx, i in enumerate(data):
            buff = []

            for j, item in enumerate(i):
                multiplier = []
                m = []
                users = set()
                num_users = 0

                for k in item:
                    multiplier.append(int(k[action_col]))
                    m.append(k[(onehot_col)])
                    if k[0] not in users:
                        users.add(k[0])
                        num_users += 1

                buff.append(list(np.hstack([sum(multiplier), num_users,
                                            np.dot(np.array(multiplier).T, np.array(m))])))
            d.append(buff)

        a = np.array(d)
        a = np.transpose(a, (1, 0))
        for i in range(len(a)):
            for j in range(len(a[i])):
                if len(a[i][j]) != 2 + len(onehot_col):
                    a[i, j] = np.zeros(2 + len(onehot_col))
        a = a.flatten()
        d = []
        for i in a:
            for j in i:
                d.append(j)
        d = np.array(d, dtype=int)
        d = d.reshape(-1, 126, 147)
        return d, unique_ten

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
