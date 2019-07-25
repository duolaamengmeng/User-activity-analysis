import time

import dataloader
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, Adagrad
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras import regularizers


def mini_max(col):
    """
    return a scaled new col
    """
    col = np.array(col).reshape(-1, 1)
    scaler = preprocessing.StandardScaler().fit(col)
    temp = scaler.transform(col)
    temp = temp.flatten()
    return temp


def load_data(apptype_path, filePath, num_work_day, instancepath, preprocessing=False):
    if preprocessing:
        data, unique_ten = dataloader.dataloader(apptype_path, filePath, num_work_day, instancepath)
        np.save('data.npy', data)
        np.save('ten.npy', unique_ten)
    else:
        data, unique_ten = np.load('data.npy'), np.load('ten.npy')
    data = data[:, :, 1:]
    batch_size = data.shape[0]
    feature_space = data.shape[2]
    time_step = data.shape[1]
    data = data.reshape(batch_size, -1)
    lst = []
    # Individually normalize each column
    for col in data.T:
        lst.append(mini_max(col))
    lst = np.array(lst).T
    data_scaled = lst.reshape(batch_size, feature_space, time_step)
    return data_scaled, unique_ten


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


def lstm_model(epoch, data, label):
    X_train, X_test, y_train, y_test = train_test_split(
        data, label, test_size=0.25)
    epochs = epoch
    opt = Adam()
    model = Sequential()
    model.add(Dropout(0.7, input_shape=[X_train.shape[1], X_train.shape[2]]))
    model.add(LSTM(units=128,
                   return_sequences=False, bias_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.7))
    # model.add(LSTM(units=32,
    #                return_sequences=False))
    # model.add(Dropout(0.7))
    # model.add(Dropout(rate = 0.5, noise_shape=None, seed=None))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=['binary_accuracy'])
    model.fit(X_train, y_train, batch_size=32, shuffle=True,
              epochs=epochs, validation_data=(X_test, y_test))
    print(X_test.shape)
    prediction = model.predict(X_test)
    prediction = (prediction > 0.5)
    print('accuracy score: {}'.format(accuracy_score(y_test, prediction)))
    # print(np.array(*list(np.array([[i,j] for i in prediction for j in y_test]).T), sep='\n'))


if __name__ == '__main__':
    t = time.time()
    data, unique_ten = load_data('C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\appType.csv',
                                 'C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\data_all.npy', 125,
                                 'C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\instance_created.csv')

    label = make_label('C:\\Users\\Administrator\\PycharmProjects\\yonyou\\data\\repaid.csv', unique_ten)

    lstm_model(1000, data, label)
    print(time.time() - t)
