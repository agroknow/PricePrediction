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


def read_cleanse():
    df = pd.read_csv("food_dataset.csv")

    dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
    dfevoo = dfevoo[dfevoo['country'] == 'greece']
    dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
    dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
        by='priceStringDate')
    dfevoo = pd.DataFrame(dfevoo)
    dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
    dfevoo['date'] = pd.to_datetime(dfevoo['priceStringDate'])
    dfevoo['year'] = pd.DatetimeIndex(dfevoo['date']).year
    dfevoo['month'] = pd.DatetimeIndex(dfevoo['date']).month
    dfevoo['day'] = pd.DatetimeIndex(dfevoo['date']).day

    dfevoo = dfevoo[dfevoo['date'] < test_date]

    columns = ['year', 'month', 'day', 'price']
    dataset = dfevoo[columns]
    return dataset


dataset = read_cleanse()

dataset = dataset.dropna()
target = 'price'

train_perc = 0.9

train_dataset = dataset[:int(len(dataset) * train_perc)]  # .sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()

print(train_stats)

train_stats.pop(target)
train_stats = train_stats.transpose()

train_labels = train_dataset.pop(target)
test_labels = test_dataset.pop(target)


def norm(x, stats):
    return (x - stats['mean']) / stats['std']


normed_train_data = norm(train_dataset, train_stats)
normed_test_data = norm(test_dataset, train_stats)



#normed_train_data=np.reshape(normed_train_data, (normed_train_data.shape[0], normed_train_data.shape[1], 1))
def build_model():
    t_model = Sequential()
    # t_model.add(LSTM(units=50, return_sequences=True, input_shape=(1, normed_train_data.shape[1],1)))
    #
    # # Dropout layer is added to avoid over-fitting, which is a phenomenon where a machine learning model performs better on the training data compared to the test data
    # t_model.add(Dropout(0.2))
    #
    # #add three more LSTM and dropout layers to our model
    # t_model.add(LSTM(units=100, return_sequences=True))
    # t_model.add(Dropout(0.2))
    #
    # t_model.add(LSTM(units=100, return_sequences=True))
    # t_model.add(Dropout(0.2))
    #
    # t_model.add(LSTM(units=100))
    #t_model.add(Dropout(0.2))
    t_model.add(Dense(64, activation="relu", input_shape=[len(train_dataset.keys())]))
    t_model.add(Dense(64, activation="relu"))
    t_model.add(Dense(1))
    t_model.compile(loss='mean_squared_error',
                    optimizer='nadam',
                    metrics=['mean_absolute_error', 'mean_squared_error'])
    return t_model


model = build_model()

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(normed_train_data, train_labels, epochs=500,
                    validation_split=0.2, verbose=1, callbacks=[early_stop])


test_df = pd.read_csv("food_dataset.csv")

dfevoo = test_df[test_df['product'] == 'extra virgin olive oil (up to 0,8°)']
dfevoo = dfevoo[dfevoo['country'] == 'greece']
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
dfevoo['date'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo['year'] = pd.DatetimeIndex(dfevoo['date']).year
dfevoo['month'] = pd.DatetimeIndex(dfevoo['date']).month
dfevoo['day'] = pd.DatetimeIndex(dfevoo['date']).day

dfevoo = dfevoo[dfevoo['date'] >= test_date]

columns = ['year', 'month', 'day', 'price']
test_df = dfevoo[columns]
unknown_stats = test_df.describe()

print(unknown_stats)

unknown_stats.pop(target)
unknown_stats = unknown_stats.transpose()

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
