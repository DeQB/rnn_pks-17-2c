import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
from numpy import array

key="ae6f21b55e3e69fc7ec16d0fcb64bce08a3eba3f"
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL.csv')

df=pd.read_csv('AAPL.csv')
df1=df.reset_index()['close']

plt.plot(df1)
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=2,batch_size=64,verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

print(math.sqrt(mean_squared_error(y_train,train_predict)))
print(math.sqrt(mean_squared_error(ytest,test_predict)))

look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

len(test_data)

x_input=test_data[340:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output = []
n_steps = 100
i = 0

while (i < 30):

     if (len(temp_input) > 100):
         x_input = np.array(temp_input[1:])
         print("{} day input {}".format(i, x_input))
         x_input = x_input.reshape(1, -1)
         x_input = x_input.reshape((1, n_steps, 1))
         print(x_input)
         yhat = model.predict(x_input, verbose=0)
         print("{} day output {}".format(i, yhat))
         temp_input.extend(yhat[0].tolist())
         temp_input = temp_input[1:]
         # print(temp_input)
         lst_output.extend(yhat.tolist())
         i = i + 1
     else:
         x_input = x_input.reshape((1, n_steps, 1))

         yhat = model.predict(x_input, verbose=0)
         temp_input.extend(yhat[0].tolist())
         lst_output.extend(yhat.tolist())
         i = i + 1

day_new=np.arange(1,101)
day_pred=np.arange(101,131)

plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.show()

df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
plt.show()

df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
plt.show()
