"""
This Source Code Form is subject to the terms of the Mozilla
Public License, v. 2.0. If a copy of the MPL was not distributed
with this file, You can obtain one at https://github.com/DeQB/rnn_pks-17-2c.
© Кобелев Д. Е.
"""

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from math import sqrt
from matplotlib import pyplot
from numpy import array
from numpy import zeros
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D


# функция синтаксического анализа даты и времени для загрузки набора данных
def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')


# преобразование временных рядов в контролируемую учебную задачу
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # входная последовательность (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # последовательность прогноза (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # sum
    agg = concat(cols, axis=1)
    agg.columns = names
    # убрать строки со значениями NaN
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# создание дифференцированного ряда
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# преобразование серий в обучающие и тестовые наборы для контролируемого обучения
def prepare_data(series, n_test, n_lag, n_seq):
    # извлечение необработанных значений
    raw_values = series.values
    # преобразование данных в стационарные
    diff_series = difference(raw_values, n_lag)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), n_lag)
    # пересчет значений -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    # scaled_values = scaled_values.reshape(len(scaled_values), n_lag)
    # преобразование в контролируемую учебную задачу X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # разделение на train и test наборы
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


def prepare_data3d(series, n_test, n_lag, n_seq):
    # извлечение необработанных значений
    raw_values = series.values
    # преобразование данных в стационарные
    diff_series = difference(raw_values, n_lag)
    diff_values = zeros((len(diff_series), raw_values.shape[1]))
    for i in range(len(diff_series)):
        diff_values[i] = diff_series.values[i]

    # diff_values = diff_values.reshape(len(diff_values), 3)
    # масштабирование значений до to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    # scaled_values = scaled_values.reshape(len(scaled_values), 1)

    # преобразование в контролируемую учебную задачу X, y
    # supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    # supervised_values = supervised.values
    supervised_values = scaled_values
    # разделение на train и test наборы
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


# приспособление сети LSTM к обучающим данным
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # дизайн сети
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # построение сети
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model


def fit_lstm_stack(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # дизайн сети
    model = Sequential()
    model.add(
        LSTM(n_neurons, return_sequences=True, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # построение сети
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model


def fit_lstm_bidir(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # дизайн сети
    model = Sequential()
    model.add(Bidirectional(LSTM(n_neurons, stateful=True), batch_input_shape=(n_batch, X.shape[1], X.shape[2])))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # построение сети
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model


def fit_lstm_CNN(train, n_lag, n_seq, n_steps, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, subsequences, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape((X.shape[0], n_seq, n_steps, X.shape[1]))
    # дизайн сети
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'),
                              batch_input_shape=(1, None, n_steps, 1)))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=1))) # 2
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # построение сети
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model


# строительство прогноза с помощью LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # создание прогноза
    forecast = model.predict(X, batch_size=n_batch)
    # преобразование в массив
    return [x for x in forecast[0, :]]


# модель персистентности.оценка
def make_forecasts(model, n_batch, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X = test[i, 0:n_lag]
        # создание проноза
        forecast = forecast_lstm(model, X, n_batch)
        # зранение прогноза
        forecasts.append(forecast)
    return forecasts


# строительство прогноза с помощью LSTM,
def forecast_lstm_CNN(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, 1, len(X))
    # создание проноза
    forecast = model.predict(X, batch_size=n_batch)
    # преобразование в массив
    return [x for x in forecast[0, :]]


# evaluate the persistence model
def make_forecasts_CNN(model, n_batch, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X = test[i, 0:n_lag]
        # создание проноза
        forecast = forecast_lstm_CNN(model, X, n_batch)
        # хранение прогноза
        forecasts.append(forecast)
    return forecasts


def make_forecasts_fin(model, n_batch, train, n_lag, n_seq):
    forecasts = list()

    X = train[len(train) - 1, n_lag + 2:]
    # создание проноза
    forecast = forecast_lstm(model, X, n_batch)
    # зранение прогноза
    forecasts.append(forecast)

    return forecasts


# инвертированный дифференцированный прогноз
def inverse_difference(last_ob, forecast):
    # инвертировать первый прогноз
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # распространение прогноза с использованием инвертированного первого значения
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i - 1])
    return inverted


# обратное преобразование данных в прогнозах
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # создание массива из прогноза
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # инвертировать масштабирование
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # инвертированное дифференцирование
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # хранение\добавление
        inverted.append(inv_diff)
    return inverted


# оценка СКО(RMSE) для каждого прогнозного шага
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i + 1), rmse))


# построение прогнозов
def plot_forecasts(series, forecasts, n_test, name):

    pyplot.plot(series.values)
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
       # pyplot.xlim(1, 12)
        pyplot.plot(xaxis, yaxis, color='red')

   #pyplot.xlim(1, 12)
    pyplot.savefig(f'img/{name}')
    pyplot.show()

