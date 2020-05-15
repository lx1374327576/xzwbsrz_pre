import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import preprocessing
from tensorflow.keras import datasets, layers, models
from keras.models import Sequential
from keras.layers.embeddings import Embedding
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
df1 = pd.read_csv("/content/drive/My Drive/pre_data/train_preliminary/user.csv")
df2 = pd.read_csv("/content/drive/My Drive/pre_data/train_preliminary/ad.csv")
df3 = pd.read_csv("/content/drive/My Drive/pre_data/train_preliminary/click_log.csv")
df_tmp = pd.merge(df3, df1, left_on='user_id', right_on='user_id')
df = pd.merge(df_tmp, df2, left_on='creative_id', right_on='creative_id')
order = ['user_id', 'time', 'creative_id', 'click_times', 'ad_id', 'product_category', 'advertiser_id']
df = df[order]
df.sort_values(['user_id', 'time'], inplace=True)
df.reset_index(drop=True, inplace=True)
x_tmp = np.array(df['creative_id'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((29, 1))))
print(x_tmp.shape)
x_tmp = x_tmp.reshape((300828, 100))
model = Sequential()
model.add(Embedding(4445719, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((30082800, 5))
x_tmp = pd.DataFrame(y_tmp[:30082771, :], columns=['creative_id_0', 'creative_id_1', 'creative_id_2', 'creative_id_3', 'creative_id_4'])
X = df[['user_id', 'time', 'click_times']]
X = pd.concat([X, x_tmp], axis=1)
x_tmp = np.array(df['ad_id'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((29, 1))))
print(x_tmp.shape)
x_tmp = x_tmp.reshape((300828, 100))
model = Sequential()
model.add(Embedding(3812201, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((30082800, 5))
x_tmp = pd.DataFrame(y_tmp[:30082771, :], columns=['ad_id_0', 'ad_id_1', 'ad_id_2', 'ad_id_3', 'ad_id_4'])
X = pd.concat([X, x_tmp], axis=1)
x_tmp = np.array(df['product_category'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((29, 1))))
print(x_tmp.shape)
x_tmp = x_tmp.reshape((300828, 100))
model = Sequential()
model.add(Embedding(20, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((30082800, 5))
x_tmp = pd.DataFrame(y_tmp[:30082771, :], columns=['product_category_0', 'product_category_1',
                                                   'product_category_2', 'product_category_3', 'product_category_4'])
X = pd.concat([X, x_tmp], axis=1)
x_tmp = np.array(df['advertiser_id'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((29, 1))))
x_tmp = x_tmp.reshape((300828, 100))
model = Sequential()
model.add(Embedding(62966, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((30082800, 5))
x_tmp = pd.DataFrame(y_tmp[:30082771, :], columns=['advertiser_id_0', 'advertiser_id_1',
                                                   'advertiser_id_2', 'advertiser_id_3', 'advertiser_id_4'])
X = pd.concat([X, x_tmp], axis=1)
print('data prepare success')


def check_matrix(x):
    if x.shape[0] >= 30:
        return x.iloc[:30, :]
    else:
        tmp1 = np.zeros((30 - x.shape[0], x.shape[1]))
        tmp1 = pd.DataFrame(tmp1)
        tmp1.columns = x.columns
        return pd.concat([x, tmp1], axis=0)


X = X.groupby('user_id').apply(check_matrix)
X = X.reset_index(drop=True)
X = X.drop('user_id', axis = 1)
X = preprocessing.MinMaxScaler().fit_transform(X)
X = np.array(X).reshape((-1, 30, 22))
print('data transfer success')

X_train, X_test = X[:800000, :, :], X[800000:, :, :]
Y2_train, Y2_test = df1.iloc[:800000, 2:3], df1.iloc[800000:, 2:3]
model = models.Sequential()
model.add(layers.Masking(mask_value=0., input_shape=(20, 22)))
model.add(layers.LSTM(32))
model.add(layers.Dense(1))
model.summary()
model.compile(loss='MSE', optimizer='adam')
model.fit(X_train, Y2_train, epochs = 100, batch_size=32)
model.evaluate(X_test, Y2_test)

