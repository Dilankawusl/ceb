import tensorflow as tf
import os
import numpy as np

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


file = r'finaldataf.csv'
df = pd.read_csv(file, parse_dates=[0])

df.index = df['Date']



df1=df[['T1', 'hour', 'day_of_month', 'day_of_week','month']]
df2=df[['T2', 'hour', 'day_of_month', 'day_of_week','month']]
df3=df[['T3', 'hour', 'day_of_month', 'day_of_week','month']]
df4=df[['T4', 'hour', 'day_of_month', 'day_of_week','month']]

temp1 = df1['T1']
temp2 = df2['T2']
temp3 = df3['T3']
temp4 = df4['T4']


def df_to_X_y(df1, window_size=12):
  df_as_np = df1.to_numpy()
  X = []
  y = []
  for i in range(len(df_as_np)-window_size):
    row = [[a] for a in df_as_np[i:i+window_size]]
    X.append(row)
    label = df_as_np[i+window_size]
    y.append(label)
  return np.array(X), np.array(y)


WINDOW_SIZE = 12
X1, y1 = df_to_X_y(temp1, WINDOW_SIZE)
X2, y2 = df_to_X_y(temp2, WINDOW_SIZE)
X3, y3 = df_to_X_y(temp3, WINDOW_SIZE)
X4, y4 = df_to_X_y(temp4, WINDOW_SIZE)


X_train1, y_train1 = X1[0:], y1[0:]
X_val1, y_val1 = X1[12000:15500], y1[12000:15500]
X_test1, y_test1 = X1[15500:], y1[15500:]

X_train2, y_train2 = X2[0:], y2[0:]
X_val2, y_val2 = X2[12000:15500], y2[12000:15500]
X_test2, y_test2 = X2[15500:], y2[15500:]

X_train3, y_train3 = X3[0:], y3[0:]
X_val3, y_val3 = X3[12000:15500], y3[12000:15500]
X_test3, y_test3 = X3[15500:], y3[15500:]

X_train4, y_train4 = X4[0:], y4[0:]
X_val4, y_val4 = X4[12000:15500], y4[12000:15500]
X_test4, y_test4 = X4[15500:], y4[15500:]



model1 = Sequential()
model1.add(InputLayer((12, 1)))
model1.add(LSTM(64))
model1.add(Dense(8, 'relu'))
model1.add(Dense(1, 'linear'))

model2 = Sequential()
model2.add(InputLayer((12, 1)))
model2.add(LSTM(64))
model2.add(Dense(8, 'relu'))
model2.add(Dense(1, 'linear'))

model3 = Sequential()
model3.add(InputLayer((12, 1)))
model3.add(LSTM(64))
model3.add(Dense(8, 'relu'))
model3.add(Dense(1, 'linear'))

model4 = Sequential()
model4.add(InputLayer((12, 1)))
model4.add(LSTM(64))
model4.add(Dense(8, 'relu'))
model4.add(Dense(1, 'linear'))


cp1 = ModelCheckpoint('model1/', save_best_only=True)
model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
cp2 = ModelCheckpoint('model2/', save_best_only=True)
model2.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
cp3 = ModelCheckpoint('model3/', save_best_only=True)
model3.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
cp4 = ModelCheckpoint('model4/', save_best_only=True)
model4.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])


model1.fit(X_train1, y_train1, validation_data=(X_val1, y_val1), epochs=60, callbacks=[cp1])
model2.fit(X_train2, y_train2, validation_data=(X_val2, y_val2), epochs=60, callbacks=[cp2])
model3.fit(X_train3, y_train3, validation_data=(X_val3, y_val3), epochs=60, callbacks=[cp3])
model4.fit(X_train4, y_train4, validation_data=(X_val4, y_val4), epochs=60, callbacks=[cp4])

tf.keras.models.save_model(model1,'E:/pro/thul.hdf5')
tf.keras.models.save_model(model2,'E:/pro/veya.hdf5')
tf.keras.models.save_model(model3,'E:/pro/kuru.hdf5')
tf.keras.models.save_model(model4,'E:/pro/kega.hdf5')