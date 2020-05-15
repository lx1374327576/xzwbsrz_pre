import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb
import csv
from sklearn import preprocessing
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers.core import Masking
from keras.layers.embeddings import Embedding
df1 = pd.read_csv("/home/aistudio/data/data35082/user.csv")
df2 = pd.read_csv("/home/aistudio/data/data35082/ad.csv")
df3 = pd.read_csv("/home/aistudio/data/data35082/click_log.csv")
df4 = pd.read_csv("/home/aistudio/data/data35082/test_ad.csv")
df5 = pd.read_csv("/home/aistudio/data/data35082/test_click_log.csv")
df2 = pd.concat([df2, df4])
df2.sort_values('creative_id', inplace=True)
df2 = df2.drop_duplicates(subset=['creative_id'], keep='first')
df3 = pd.concat([df3, df5])
df = pd.merge(df3, df2, left_on='creative_id', right_on='creative_id')
order = ['user_id', 'time', 'creative_id', 'click_times', 'ad_id', 'product_category', 'advertiser_id']
df = df[order]
df.sort_values(['user_id', 'time'], inplace=True)
df.reset_index(drop=True, inplace=True)
print('read success!')
x_tmp = np.array(df['creative_id'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((17, 1))))
print(x_tmp.shape)
x_tmp = x_tmp.reshape((-1, 100))
model = Sequential()
model.add(Embedding(5000000, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((-1, 5))
x_tmp = pd.DataFrame(y_tmp[:63668283, :], columns=['creative_id_0', 'creative_id_1', 'creative_id_2', 'creative_id_3', 'creative_id_4'])
X = df[['user_id', 'time', 'click_times']]
X = pd.concat([X, x_tmp], axis=1)
x_tmp = np.array(df['ad_id'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((17, 1))))
print(x_tmp.shape)
x_tmp = x_tmp.reshape((-1, 100))
model = Sequential()
model.add(Embedding(4000000, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((-1, 5))
x_tmp = pd.DataFrame(y_tmp[:63668283, :], columns=['ad_id_0', 'ad_id_1', 'ad_id_2', 'ad_id_3', 'ad_id_4'])
X = pd.concat([X, x_tmp], axis=1)
x_tmp = np.array(df['product_category'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((17, 1))))
print(x_tmp.shape)
x_tmp = x_tmp.reshape((-1, 100))
model = Sequential()
model.add(Embedding(20, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((-1, 5))
x_tmp = pd.DataFrame(y_tmp[:63668283, :], columns=['product_category_0', 'product_category_1',
                                                   'product_category_2', 'product_category_3', 'product_category_4'])
X = pd.concat([X, x_tmp], axis=1)
x_tmp = np.array(df['advertiser_id'])
x_tmp = x_tmp.reshape((-1, 1))
x_tmp = np.vstack((x_tmp, np.zeros((17, 1))))
x_tmp = x_tmp.reshape((-1, 100))
model = Sequential()
model.add(Embedding(70000, 5, input_length=100))
model.compile(loss='MSE', optimizer='adam')
y_tmp = model.predict(x_tmp, verbose=1)
y_tmp = y_tmp.reshape((-1, 5))
x_tmp = pd.DataFrame(y_tmp[:63668283, :], columns=['advertiser_id_0', 'advertiser_id_1',
                                                   'advertiser_id_2', 'advertiser_id_3', 'advertiser_id_4'])
X = pd.concat([X, x_tmp], axis=1)
print('prepare success!')


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
X = np.array(X).reshape((1900000, 30, 22))
print('transfer success!')


X_train, X_test = X[:900000, :, :], X[900000:, :, :]
Y2_train = df1.iloc[:900000, 2:3]
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(30, 22)))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='MSE', optimizer='adam')
model.fit(X_train, Y2_train, nb_epoch=2)

Y1_train = df1.iloc[:900000, 1:2]
Y1_train = pd.get_dummies(Y1_train['age'])
model2 = Sequential()
model2.add(Masking(mask_value=0., input_shape=(30, 22)))
model2.add(LSTM(32))
model2.add(Dense(30, activation='relu'))
model2.add(Dense(10, activation='softmax'))
model2.compile(loss='MSE', optimizer='adam')
model2.fit(X_train, Y1_train, nb_epoch=2)

Y = model2.predict(X_train)
ans = 0
for i in range(900000):
    if np.argmax(Y[i]) == df1.iloc[i, 1] - 1:
        ans += 1
print('age accurency:', ans/900000)

Y = model.predict(X_train)
Y = np.round(Y)
ans = 0
for i in range(900000):
    if Y[i] == Y2_train.iloc[i, 0]:
        ans += 1
print('gender accurency:', ans/900000)

Y2_output = model.predict(X_test)
Y1_output = model2.predict(X_test)
Y1_output = np.argmax(Y1_output, axis=0) + 1
Y2_output = np.round(Y2_output)
with open('submission.csv', 'w', encoding='utf-8') as csvfile:
    fieldnames = ['user_id', 'predicted_age', 'predicted_gender']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(1000000):
        writer.writerow({'user_id': str(i + 3000001), 'predicted_age': str(Y1_output[i]),
                         'predicted_gender': str(Y2_output[i])})
