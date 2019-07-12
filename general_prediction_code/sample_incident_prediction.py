from __future__ import print_function

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

data = json.loads(open('sample.json', 'r').read())

threshold = 50
products = []
for bucket in data['aggregations']['sterms#years']['buckets']:
    if bucket['doc_count'] > threshold:

        for subagg in bucket['date_histogram#subaggregation']['buckets']:
            products.append([bucket['key'], subagg['key_as_string'], subagg['doc_count']])
print(products)
# END OF FETCH MONTH BASIS


categorical_columns = ['product']
columns = ['year', 'month', 'count']
df = pd.DataFrame(products, columns=['product', 'date', 'count'])
df['date'] = pd.to_datetime(df['date'])
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month

print(df.head())
df = df.sort_values(by='date')
df = df[df['product'] == 'nuts, nut products and seeds']
df = df[columns]

# # Categorical boolean mask
# # categorical_feature_mask = df.dtypes == object
# # # filter categorical columns using mask and turn it into a list
# # categorical_cols = df.columns[categorical_feature_mask].tolist()
# # # import labelencoder
# # from sklearn.preprocessing import LabelEncoder
# #
# # # instantiate labelencoder object
# # le = LabelEncoder()
# # # apply le on categorical feature columns
# # df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
# # print(df[categorical_cols].head(10))

kc_data_org = df

kc_data = pd.DataFrame(kc_data_org, columns=columns)
label_col = 'count'

print(kc_data.describe())


def train_validate_test_split(df, train_part=.6, validate_part=.2, test_part=.2, seed=None):
    np.random.seed(seed)
    total_size = train_part + validate_part + test_part
    train_percent = train_part / total_size
    validate_percent = validate_part / total_size
    test_percent = test_part / total_size
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = perm[:train_end]
    validate = perm[train_end:validate_end]
    test = perm[validate_end:]
    return train, validate, test


train_size, valid_size, test_size = (70, 20, 10)
kc_train, kc_valid, kc_test = train_validate_test_split(kc_data,
                                                        train_part=train_size,
                                                        validate_part=valid_size,
                                                        test_part=test_size,
                                                        seed=2017)

kc_y_train = kc_data.loc[kc_train, [label_col]]
kc_x_train = kc_data.loc[kc_train, :].drop(label_col, axis=1)
kc_y_valid = kc_data.loc[kc_valid, [label_col]]
kc_x_valid = kc_data.loc[kc_valid, :].drop(label_col, axis=1)
kc_y_test = kc_data.loc[kc_test, [label_col]]
kc_x_test = kc_data.loc[kc_test, :].drop(label_col, axis=1)

print('Size of training set: ', len(kc_x_train))
print('Size of validation set: ', len(kc_x_valid))
print('Size of test set: ', len(kc_test), '(not converted)')


def norm_stats(df1, df2):
    dfs = df1.append(df2)
    minimum = np.min(dfs)
    maximum = np.max(dfs)
    mu = np.mean(dfs)
    sigma = np.std(dfs)
    return (minimum, maximum, mu, sigma)


def z_score(col, stats):
    m, M, mu, s = stats
    df = pd.DataFrame()
    for c in col.columns:
        df[c] = (col[c] - mu[c]) / s[c]
    return df


stats = norm_stats(kc_x_train, kc_x_valid)
arr_x_train = np.array(z_score(kc_x_train, stats))
arr_y_train = np.array(kc_y_train)
arr_x_valid = np.array(z_score(kc_x_valid, stats))
arr_y_valid = np.array(kc_y_valid)
arr_x_test = np.array(z_score(kc_x_test, stats))
arr_y_test = np.array(kc_y_test)

print('Training shape:', arr_x_train.shape)
print('Training samples: ', arr_x_train.shape[0])
print('Validation samples: ', arr_x_valid.shape[0])


def basic_model_3(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(80, activation="tanh", kernel_initializer='normal', input_shape=(x_size,)))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(120, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1(0.01), bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(20, activation="relu", kernel_initializer='normal',
                      kernel_regularizer=regularizers.l1_l2(0.01), bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size))
    t_model.compile(
        loss='mean_squared_error',
        optimizer='nadam',
        metrics=[metrics.mae])
    return (t_model)


model = basic_model_3(arr_x_train.shape[1], arr_y_train.shape[1])
model.summary()

epochs = 500
batch_size = 32

print('Epochs: ', epochs)
print('Batch size: ', batch_size)

keras_callbacks = [
    EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)
]

history = model.fit(arr_x_train, arr_y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    shuffle=True,
                    verbose=1,  # Change it to 2, if wished to observe execution
                    validation_data=(arr_x_valid, arr_y_valid),
                    callbacks=keras_callbacks)

train_score = model.evaluate(arr_x_train, arr_y_train, verbose=1)
valid_score = model.evaluate(arr_x_valid, arr_y_valid, verbose=1)

print('Train MAE: ', round(train_score[1], 4), ', Train Loss: ', round(train_score[0], 4))
print('Val MAE: ', round(valid_score[1], 4), ', Val Loss: ', round(valid_score[0], 4))


def plot_hist(h, xsize=6, ysize=10):
    # Prepare plotting
    fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [xsize, ysize]
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True)

    # summarize history for MAE
    plt.subplot(211)
    plt.plot(h['mean_absolute_error'])
    plt.plot(h['val_mean_absolute_error'])
    plt.title('Training vs Validation MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # summarize history for loss
    plt.subplot(212)
    plt.plot(h['loss'])
    plt.plot(h['val_loss'])
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot it all in IPython (non-interactive)
    plt.draw()
    plt.show()

    return


plot_hist(history.history, xsize=8, ysize=12)

Xt = model.predict(arr_x_test)
plt.plot(Xt)
# plt.legend(['Predictions', 'Actual'], loc='upper left')
# plt.show()
plt.plot(arr_y_test)
# plt.legend(['Actual on Test'], loc='upper left')
plt.show()
#
# print(arr_x_valid)
#
# Xt = model.predict(arr_x_valid)
# plt.plot(Xt)
# plt.legend(['Predictions on Validation'], loc='upper left')
# plt.show()
# plt.plot(arr_y_valid)
# plt.legend(['Actual on Validation'], loc='upper left')
# plt.show()
