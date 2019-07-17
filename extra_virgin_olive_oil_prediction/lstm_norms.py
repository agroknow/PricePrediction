import requests
import json
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# https://stackoverflow.com/questions/42532386/how-to-work-with-multiple-inputs-for-lstm-in-keras
# interesting: https://datascience.stackexchange.com/questions/17024/rnns-with-multiple-features


test_date = '2019-01-01'
train_split = 0.9
dfevoo = None
columns = ['year', 'month', 'day', 'date', 'price']


def read_cleanse(train=True, dfevoo=None):
    if train:
        df = pd.read_csv("food_dataset.csv")

        dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
        # dfevoo = df[df['product'].str.contains("μήλα")]
        dfevoo = dfevoo[dfevoo['country'] == 'greece']
        psd = pd.to_datetime(dfevoo['priceStringDate'])
        dfevoo['priceStringDate'] = psd
        dfevoo = dfevoo.sort_values(by='priceStringDate')
        dfevoo = pd.DataFrame(dfevoo)
        # dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
        dfevoo['date'] = pd.to_datetime(dfevoo['priceStringDate'])
        dfevoo['year'] = pd.DatetimeIndex(dfevoo['date']).year
        dfevoo['month'] = pd.DatetimeIndex(dfevoo['date']).month
        dfevoo['day'] = pd.DatetimeIndex(dfevoo['date']).day

        # dfevoo['year'] = pd.to_timedelta(dfevoo['priceStringDate'], unit='ns').dt.total_seconds().astype(int)

        # dfevoo = dfevoo.sample(frac=1)

        dataset = dfevoo[columns]
        # dataset = dataset[:int(len(dataset) * train_split)]
        dataset = dataset[dataset['date'] < test_date]
        return dataset, dfevoo
    else:
        dataset = dfevoo[columns]
        # dataset = dataset[int(len(dataset) * train_split):]
        dataset = dataset[dataset['date'] >= test_date]
        return dataset


dataset, dfevoo = read_cleanse()
dataset.pop('date')
dataset = dataset.dropna()
target = 'price'

train_perc = 1.0

train_dataset = dataset[:int(len(dataset) * train_perc)]  # .sample(frac=0.8, random_state=0)
# test_dataset = dataset.drop(train_dataset.index)


def dense_model():
    train_stats = dataset.describe()

    print(train_stats)

    train_stats.pop(target)
    train_stats = train_stats.transpose()

    train_labels = train_dataset.pop(target)
    # test_labels = test_dataset.pop(target)


    def norm(x, stats):
        return (x - stats['mean']) / stats['std']


    normed_train_data = norm(train_dataset, train_stats)
    # normed_test_data = norm(test_dataset, train_stats)


    # normed_train_data=np.reshape(normed_train_data, (normed_train_data.shape[0], normed_train_data.shape[1], 1))
    def build_model():
        t_model = Sequential()
        t_model.add(Dense(512, activation="relu", input_shape=[len(train_dataset.keys())]))
        t_model.add(Dropout(0.1))
        t_model.add(Dense(256, activation="relu"))
        t_model.add(Dropout(0.1))
        t_model.add(Dense(128, activation="relu"))
        t_model.add(Dropout(0.1))
        t_model.add(Dense(64, activation="relu"))
        t_model.add(Dropout(0.1))
        t_model.add(Dense(16, activation="relu"))
        t_model.add(Dense(1))
        t_model.compile(loss='mean_squared_error',
                        optimizer='nadam',
                        metrics=['mean_absolute_error', 'mean_squared_error'])
        return t_model


    model = build_model()

    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=20)

    history = model.fit(normed_train_data, train_labels, epochs=500,
                        validation_split=0.4, verbose=1, callbacks=[early_stop])
    test_df = read_cleanse(train=False, dfevoo=dfevoo)
    test_df.pop('date')

    unknown_labels = test_df.pop(target)

    normed_unknown_data = norm(test_df, train_stats)
    unknown_predictions = model.predict(normed_unknown_data).flatten()

    print(unknown_predictions)
    print(unknown_labels)

    # unknown_predictions = unknown_predictions - 40
    print(unknown_predictions)

    plt.plot(unknown_predictions)
    plt.plot(unknown_labels.values)
    plt.title('price prediction')
    plt.legend(['Predictions', 'Actual'], loc='upper left')
    plt.show()



dense_model()
