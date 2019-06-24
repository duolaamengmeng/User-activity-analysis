from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csc_matrix
from IPython.display import clear_output


class DataLoader:
    def __init__(self, filename, time_col, onehot_features,
                 features, ten_col, error,
                 error_col, data_split=True, sample_size=500,
                 shuffle=True, batch_size=64):

        self.df = pd.read_table(filename)
        self.time_col = time_col
        self.onehot_features = onehot_features
        self.features = features
        self.ten_col = ten_col
        self.error_col = error_col
        self.error = error
        self.data_split = data_split
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.batch_size = batch_size

    def create_error(self):
        for i, item in enumerate(self.df[self.error_col]):
            if item in self.error:
                self.df[self.error_col][i] = 'error'
        return self.df

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
        one_hot_df = [self.find_index(i) for i in self.onehot_features]
        one_hot_ex = [self.features.index(i) for i in self.onehot_features]
        # not one hot treated variables' index
        exclude = [i for i in range(len(self.features)) if i not in one_hot_ex]

        # one hot encoding
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(data[:, one_hot_df])
        one_hot_value = enc.transform(data[:, one_hot_df])
        data = hstack([data[:, exclude].astype(float), one_hot_value])
        return data

    def time(self, array):
        # find the column index of time feature
        time_col = self.find_index(self.time_col)

        # determine the time step list
        time_bin = self.find_time_bin()
        data = []

        # iterate through each timestep, create a new dimension for time step
        for indx, i in enumerate(time_bin):
            clear_output(wait=True)
            print('step 1: iteration {}  out of {}'.format(indx, len(time_bin)))
            buff = []
            for j, item in enumerate(array):
                if item.toarray()[0][time_col] == i:
                    buff.append(array[j, :])
            data.append(buff)
        return data

    def make3dts(self, lst):
        ten = self.find_index(self.ten_col)
        unique_ten = list(set(self.df.iloc[:, ten]))
        for j_index, j in enumerate(lst):  # iterate time step

            clear_output(wait=True)
            print('step 2: iteration {}  out of {}'.format(j_index, len(lst)))

            cpp = []
            for i in unique_ten:  # iterate unique companies
                cp = []
                for k_index, k in enumerate(j):  # iterate each element in time step
                    if int(k.toarray()[0][ten]) == int(i):
                        cp.append(lst[j_index][k_index])
                    else:
                        cp.append(csc_matrix(np.zeros([k.shape[1]])))

                cpp.append(cp)
            lst[j_index] = cpp
        return lst

    def sum_all(self, data):
        d = []
        for indx, i in enumerate(data):
            buff = []
            for j, item in enumerate(i):
                multiplier = []
                m = []
                for k in item:
                    multiplier.append(k.toarray()[0][1])
                    m.append(k.toarray()[0])
                buff.append(np.dot(np.array(multiplier).T, np.array(m)[:, 4:]))
            d.append(buff)
        return d

    def batch_generator(self):
        """generate batch given self.batch_size and numpy array or DataFrame
        usage: next() to generate the next batch"""
        all_data = self.all_data()
        all_data = [np.array(d) for d in all_data]
        data_size = all_data[0].shape[0]
        print("data_size: ", data_size)
        if self.shuffle:
            p = np.random.permutation(data_size)
            all_data = [d[p] for d in all_data]

        batch_count = 0
        while True:
            if batch_count * self.batch_size + self.batch_size > data_size:
                batch_count = 0
                if self.shuffle:
                    p = np.random.permutation(data_size)
                    all_data = [d[p] for d in all_data]
            start = batch_count * self.batch_size
            end = start + self.batch_size
            batch_count += 1
            yield [d[start: end] for d in all_data]

    def all_data(self):
        if self.data_split:
            self.df = self.df.sample(n=self.sample_size)
        data = self.pre_processing()
        matrix = data.tocsc()

        matrix = self.time(matrix)
        a = self.make3dts(matrix)

        a = np.array(a)
        a = a.transpose(1, 0)
        b = self.sum_all(a)
        return b