from tkinter import *
import pandas as pd

def open_window():
    top=Toplevel()
    top.title('Models')
    top.geometry('650x500')
    top.resizable(False, False)


    readSeries = pd.read_csv("series.csv", index_col=0)
    series = readSeries['bidopen']

    readValues = pd.read_csv("values.csv", index_col=0)
    values = readValues['tickqty']

    row_count = sum(1 for row in series)
    from defs_neuro import prepare_data, fit_lstm, make_forecasts, inverse_transform, evaluate_forecasts, plot_forecasts

    def model1():

        n_lag = 1
        n_seq = 1
        n_test = 7
        n_epochs = 200
        n_batch = 1
        n_neurons = 4

        scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

        model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

        model.reset_states()
        forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)

        forecasts = inverse_transform(series, forecasts, scaler, n_test)
        actual = [row[n_lag:] for row in test]
        actual = inverse_transform(series, actual, scaler, n_test)

        evaluate_forecasts(actual, forecasts, n_lag, n_seq)

        plot_forecasts(series[(row_count-12):(row_count)], forecasts, n_test)


    def model2():

        from defs_neuro import fit_lstm_stack

        n_lag = 1
        n_seq = 1
        n_test = 7
        n_epochs = 200
        n_batch = 1
        n_neurons = 4

        scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

        model = fit_lstm_stack(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

        model.reset_states()
        forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)

        forecasts = inverse_transform(series, forecasts, scaler, n_test)
        actual = [row[n_lag:] for row in test]
        actual = inverse_transform(series, actual, scaler, n_test)

        evaluate_forecasts(actual, forecasts, n_lag, n_seq)

        plot_forecasts(series[(row_count-12):(row_count)], forecasts, n_test)


    def model3():

        # 3 модель прогнозирования "В лоб"(слой с обратным распространением)
        from defs_neuro import fit_lstm_bidir

        n_lag = 1
        n_seq = 1
        n_test = 7
        n_epochs = 200
        n_batch = 1
        n_neurons = 3

        scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

        model = fit_lstm_bidir(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)

        model.reset_states()
        forecasts = make_forecasts(model, n_batch, test, n_lag, n_seq)

        forecasts = inverse_transform(series, forecasts, scaler, n_test - 1)
        actual = [row[n_lag:] for row in test]
        actual = inverse_transform(series, actual, scaler, n_test - 1)

        evaluate_forecasts(actual, forecasts, n_lag, n_seq)

        plot_forecasts(series[(row_count-12):(row_count)], forecasts, n_test - 1)


    def model4():

        from defs_neuro import fit_lstm_CNN, make_forecasts_CNN

        n_lag = 1
        n_test = 7
        n_epochs = 200
        n_batch = 1
        n_neurons = 4
        n_seq = 1
        n_steps = 1

        scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)

        model = fit_lstm_CNN(train, n_lag, n_seq, n_steps, n_batch, n_epochs, n_neurons)

        model.reset_states()
        forecasts = make_forecasts_CNN(model, n_batch, test, n_lag, n_seq)

        forecasts = inverse_transform(series, forecasts, scaler, n_test)
        actual = [row[n_lag:] for row in test]
        actual = inverse_transform(series, actual, scaler, n_test)

        evaluate_forecasts(actual, forecasts, n_lag, n_seq)

        plot_forecasts(series[(row_count-12):(row_count)], forecasts, n_test)


    def model5():

        from numpy import vstack
        from defs_neuro import prepare_data

        n_lag = 1
        n_seq = 1
        n_test = 7
        n_epochs = 200
        n_batch = 1
        n_neurons = 4

        scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
        scaler_1, train_1, test_1 = prepare_data(series, n_test, n_lag, n_seq)
        scaler_2, train_2, test_2 = prepare_data(values, n_test, n_lag, n_seq)

        train = vstack((train_1[:, 0], train_2[:, 0], train_1[:, 1]))
        train = train.T

        model = fit_lstm(train, n_lag + 1, n_seq, n_batch, n_epochs, n_neurons)

        model.reset_states()
        test = vstack((test_1[:, 0], test_2[:, 0], test_1[:, 1]))
        test = test.T
        forecasts = make_forecasts(model, n_batch, test, n_lag + 1, n_seq)

        forecasts = inverse_transform(series, forecasts, scaler, n_test)
        actual = [row[n_lag+1:] for row in test]
        actual=inverse_transform(series, actual, scaler, n_test)

        evaluate_forecasts(actual, forecasts, n_lag, n_seq)

        plot_forecasts(series[(row_count-12):(row_count)], forecasts, n_test)


    labelSpace = Label(top, text='          ')
    labelSpace2 = Label(top, text='          ')
    labelSpace3 = Label(top, text='          ')
    labelSpace4 = Label(top, text='          ')
    labelSpace5 = Label(top, text='          ')
    labelSpace6 = Label(top, text='          ')
    labelSpace7 = Label(top, text='          ')
    labelSpace8 = Label(top, text='          ')
    labelSpace.grid(row=0, column=1)
    labelSpace2.grid(row=2, column=1)
    labelSpace3.grid(row=4, column=1)
    labelSpace4.grid(row=6, column=1)
    labelSpace5.grid(row=8, column=1)
    labelSpace6.grid(row=10, column=1)
    labelSpace7.grid(row=1, column=0, rowspan=10)
    labelSpace8.grid(row=1, column=2)



    button2 = Button(top, text='Модель 1', command=model1, width=12)
    button2.grid(row=1, column=1)

    button3=Button(top, text='Модель 2', command=model2, width=12)
    button3.grid(row=3, column=1)

    button4=Button(top, text='Модель 3', command=model3, width=12)
    button4.grid(row=5, column=1)

    button5=Button(top, text='Модель 4', command=model4, width=12)
    button5.grid(row=7, column=1)

    button6=Button(top, text='Модель 5', command=model5, width=12)
    button6.grid(row=9, column=1)

    button1 = Button(top, text='Закрыть', command=top.destroy, width=12)
    button1.grid(row=11, column=1)

    button7 = Button(top, text='Новостная модель(beta)', command=top.destroy, width=22)
    button7.grid(row=13, column=1)

    """Бинды"""
    text_info = Text(top, width=50, height=27,padx=5, pady=5, bd=2)
    text_info.grid(row=1, column=3, rowspan=12, columnspan=12)
    text_info.insert(INSERT, 'Для получения информации о модели прогнозирования нажать <cttl> + <номер модели>')
    text_info.configure(state='disabled')


    def print_info_model1(event):
        text_info = Text(top, width=50, height=27, padx=5, pady=5, bd=2)
        text_info.grid(row=1, column=3, rowspan=12, columnspan=12)
        text_info.insert(INSERT, 'Информационный текст для модели 1')
        text_info.configure(state='disabled')

    def print_info_model2(event):
        text_info = Text(top, width=50, height=27, padx=5, pady=5, bd=2)
        text_info.grid(row=1, column=3, rowspan=12, columnspan=12)
        text_info.insert(INSERT, 'Информационный текст для модели 2')
        text_info.configure(state='disabled')

    def print_info_model3(event):
        text_info = Text(top, width=50, height=27, padx=5, pady=5, bd=2)
        text_info.grid(row=1, column=3, rowspan=12, columnspan=12)
        text_info.insert(INSERT, 'Информационный текст для модели 3')
        text_info.configure(state='disabled')

    def print_info_model4(event):
        text_info = Text(top, width=50, height=27, padx=5, pady=5, bd=2)
        text_info.grid(row=1, column=3, rowspan=12, columnspan=12)
        text_info.insert(INSERT, 'Информационный текст для модели 4')
        text_info.configure(state='disabled')

    def print_info_model5(event):
        text_info = Text(top, width=50, height=27, padx=5, pady=5, bd=2)
        text_info.grid(row=1, column=3, rowspan=12, columnspan=12)
        text_info.insert(INSERT, 'Информационный текст для модели 5')
        text_info.configure(state='disabled')


    top.bind('<Control-Key-1>', print_info_model1)
    top.bind('<Control-Key-2>', print_info_model2)
    top.bind('<Control-Key-3>', print_info_model3)
    top.bind('<Control-Key-4>', print_info_model4)
    top.bind('<Control-Key-5>', print_info_model5)


