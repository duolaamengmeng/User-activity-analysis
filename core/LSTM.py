import os
import numpy as np
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Conv1D
from keras.optimizers import Adam, SGD
from keras.layers import Dropout
from keras.layers import MaxPooling1D, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import main
from sklearn.model_selection import train_test_split


def load_data(filePath, num_work_day, instancepath):
    data, unique_ten = main.process(filePath, num_work_day, instancepath)
    data = data.flatten().reshape(-1, 1)
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    d1_scaled = min_max_scaler.fit_transform(data)
    d1_scaled = d1_scaled.reshape(189, -1, 126)
    return d1_scaled, unique_ten


def make_label(filePath, unique_ten):
    repaid = pd.read_csv(filePath)
    repaid_list = repaid['instance_id'].tolist()
    instances = [int(i) for i in unique_ten]
    label = []
    for i in instances:
        if i in repaid_list:
            label.append(1)
        else:
            label.append(0)
    label = np.array(label).reshape(-1, 1)

    return label


def train_test_split(data, label):
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.10)
    return X_train, X_test, y_train, y_test


def lstm_model(epoch, X_train, X_test, y_train, y_test):
    epochs = epoch
    opt = SGD(lr=0.001)
    model = Sequential()
    model.add(LSTM(units=128,
                   return_sequences=True, input_shape=[None, 126]))

    model.add(LSTM(units=32,
                   return_sequences=False))

    # model.add(Dropout(rate = 0.5, noise_shape=None, seed=None))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['binary_accuracy'])
    model.fit(X_train, y_train, batch_size=32, shuffle=True, epochs=epochs, validation_data=(X_test, y_test))


def main():
    pass
