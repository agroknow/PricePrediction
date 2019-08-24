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

# idea_v1.0: https://www.datacamp.com/community/tutorials/lstm-python-stock-market

test_date = '2019-01-01'
train_split = 0.9
dfevoo = None
columns = ['year', 'month', 'day', 'date', 'price']


def read_cleanse(train=True, dfevoo=None):
    if train:
        df = pd.read_csv("food_dataset.csv")

        dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
        # dfevoo = dfevoo[dfevoo['country'] == 'greece']
        dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
        dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
            by='priceStringDate')
        dfevoo = pd.DataFrame(dfevoo)
        dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
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
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()

# print(train_stats)

train_stats.pop(target)
train_stats = train_stats.transpose()

train_labels = train_dataset.pop(target)
test_labels = test_dataset.pop(target)

# print(train_labels)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset.pop('price')
scaler.fit(dataset)
scaled_data = scaler.transform(train_dataset)

# print(train_dataset)
# print(scaled_data)
# print(scaled_data[0:10, 0])
# print(scaled_data[0:10])
# quit(0)
# scaled_data_train=scaler.fit_transform(train_dataset)

# Convert Training Data to Right Shape
lb = 10
normed_train_data = []
normed_test_data = []
# # execute a loop that starts from 61st record and stores all the previous 60 records to the x_train list. The 61st record is stored in the y_trainlabels list.
for i in range(lb, len(train_dataset)):
    normed_train_data.append(scaled_data[i - lb:i])
#
normed_train_data = np.array(normed_train_data)

print(normed_train_data.shape)
normed_train_data = np.reshape(normed_train_data, (normed_train_data.shape[0], normed_train_data.shape[1], 3))

# quit(0)
print(normed_train_data)


# overall_stats = dataset.describe().transpose()
# def norm(x, stats):
#     return (x - stats['mean']) / stats['std']
#
# normed_train_data = norm(train_dataset, overall_stats)
# normed_train_data = np.reshape(normed_train_data, (normed_train_data.shape[0], normed_train_data.shape[1], 3))


def build_model():
    t_model = Sequential()
    t_model.add(LSTM(units=256, return_sequences=True, input_shape=(normed_train_data.shape[1], 3)))

    # Dropout layer is added to avoid over-fitting, which is a phenomenon where a machine learning model performs better on the training data compared to the test data
    t_model.add(Dropout(0.2))

    # add three more LSTM and dropout layers to our model
    t_model.add(LSTM(units=100, return_sequences=True))
    t_model.add(Dropout(0.1))

    t_model.add(LSTM(units=100, return_sequences=True))
    t_model.add(Dropout(0.2))

    t_model.add(LSTM(units=100))
    t_model.add(Dropout(0.2))
    t_model.add(Dense(1))
    t_model.compile(loss='mean_squared_error',
                    optimizer='nadam',
                    metrics=['mean_absolute_error', 'mean_squared_error'])
    return t_model


model = build_model()

model.summary()
# print(normed_train_data)
early_stop = EarlyStopping(monitor='val_loss', patience=20)

# TODO: REVISE THIS UGLY QUICK FIX
train_labels = train_labels[:len(train_labels) - 10]
history = model.fit(normed_train_data, train_labels, epochs=500,
                    validation_split=0.2, verbose=1, callbacks=[early_stop])

test_df = read_cleanse(train=False, dfevoo=dfevoo)
test_df.pop('date')

unknown_labels = test_df.pop(target)

inputs = test_df
# print(inputs)
# print(len(inputs))
# print(inputs.shape[0])
print(test_df.head())
print(train_dataset.head())
# inputs = inputs.reshape(-1,1)
quit(0)

inputs = scaler.transform(inputs)
print(len(inputs))

X_test = []

# for i in range(lb,len(inputs)):
for i in range(lb, inputs.shape[0]):
    X_test.append(inputs[i - lb:i])
# X_test = inputs
# X_test.append(inputs[:,0])
X_test = np.array(X_test)
print(len(X_test))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 3))

unknown_predictions = model.predict(X_test)
print(unknown_predictions)

# unknown_predictions = scaler.inverse_transform(unknown_predictions)
print(len(unknown_predictions))
print(len(inputs))
print(inputs)

# unknown_predictions = unknown_predictions - 40


plt.plot(unknown_predictions)
plt.plot(unknown_labels)
plt.title('price prediction')
plt.legend(['Predictions', 'Actual'], loc='upper left')
plt.show()
