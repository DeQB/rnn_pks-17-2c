"""
This Source Code Form is subject to the terms of the Mozilla
Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at https://github.com/DeQB/rnn_pks-17-2c.
© Кобелев Д. Е.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout


def modelFinal():
    data = pd.read_csv('data/btc_data.csv', date_parser = True)
    data_training = data[data['Date']< '2020-01-01'].copy()
    data_test = data[data['Date']> '2020-01-01'].copy()
    training_data = data_training.drop(['Date', 'Adj Close'], axis = 1)
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data)
    X_train = []
    Y_train = []

    for i in range(60, training_data.shape[0]):
        X_train.append(training_data[i-60:i])
        Y_train.append(training_data[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)

    regressor = Sequential()
    regressor.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
    regressor.add(Dropout(0.3))

    regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
    regressor.add(Dropout(0.4))

    regressor.add(LSTM(units = 120, activation = 'relu'))
    regressor.add(Dropout(0.5))

    regressor.add(Dense(units =1))

    regressor.summary()

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, Y_train, epochs = 2, batch_size =50)

    past_60_days = data_training.tail(60)
    df= past_60_days.append(data_test, ignore_index = True)
    df = df.drop(['Date', 'Adj Close'], axis = 1)

    inputs = scaler.transform(df)

    X_test = []
    Y_test = []
    for i in range (60, inputs.shape[0]):
        X_test.append(inputs[i-60:i])
        Y_test.append(inputs[i, 0])

    X_test, Y_test = np.array(X_test), np.array(Y_test)
    Y_pred = regressor.predict(X_test)
    scale = 1/5.18164146e-05
    Y_test = Y_test*scale
    Y_pred = Y_pred*scale
    plt.figure(figsize=(14,5))
    plt.plot(Y_test, color = 'red', label = 'Реальная цена')
    plt.plot(Y_pred, color = 'green', label = 'Спрогнозированная цена')
    plt.title('Прогноз')
    plt.xlabel('Время')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()
    plt.savefig('img/Модель 7')

