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


def build_model():
    t_model = Sequential()
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
                    validation_split=0.2, verbose=0, callbacks=[early_stop])

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

quit(0)

print(Data.shape, train.shape, valid.shape)

# Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
N = 80
imageDir = "plots/"
count = 50
while (count < N):
    # Convert Training Data to Right Shape
    lb = count
    flag = 'result from %s' % str(count)
    x_train = []
    y_train = []
    # execute a loop that starts from 61st record and stores all the previous 60 records to the x_train list. The 61st record is stored in the y_trainlabels list.
    for i in range(lb, len(train)):
        x_train.append(scaled_data[i - lb:i, 0])
        y_train.append(scaled_data[i, 0])

    # convert both list to the numpy array before training
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # convert our data into the shape accepted by the LSTM 'three-dimensional format'.
    # The first dimension is the number of records or rows in the dataset. The second dimension is the number of time steps which is 60 while the last dimension is the number of indicators
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Training The LSTM
    model = Sequential()  # instantiate the Sequential class.

    # The first parameter to the LSTM layer is the number of neurons that we want in the layer.
    # The second parameter is return_sequences, which is set to true since we will add more layers to the model.
    # The first parameter to the input_shape is the number of time steps while the last parameter is the number of indicators.
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # Dropout layer is added to avoid over-fitting, which is a phenomenon where a machine learning model performs better on the training data compared to the test data
    model.add(Dropout(0.2))

    # add three more LSTM and dropout layers to our model
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100))
    model.add(Dropout(0.2))

    # add a dense layer at the end of the model. The number of neurons in the dense layer will be set to 1 since we want to predict a single value in the output.
    model.add(Dense(1))

    # Model Compilation
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    keras_callbacks = [
        EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)
    ]
    # Algorithm Training
    epo = 50
    fitted_model = model.fit(x_train, y_train, epochs=epo, batch_size=120, verbose=1)

    plt.plot(fitted_model.history['loss'])
    plt.plot(fitted_model.history['acc'])
    # plt.savefig('%s history.png' % flag, bbox_inches='tight')
    plt.show()

    # evaluate the model
    train_acc = model.evaluate(x_train, y_train, verbose=1)
    print('train_acc')
    print(train_acc)

    # Prepare our test inputs
    inputs = Data[len(Data) - len(valid) - lb:].values

    # Scale our test data
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    # final test input
    X_test = []
    for i in range(lb, inputs.shape[0]):
        X_test.append(inputs[i - lb:i, 0])

    # Convert our data into the three-dimensional format which can be used as input to the LSTM.
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Making Predictions
    closing_price = model.predict(X_test)
    # reverse the scaled prediction back to their actual values.
    # Use the ìnverse_transform method of the scaler object we created during training
    closing_price = scaler.inverse_transform(closing_price)

    # root mean square is a measure of the imperfection of the fit of the estimator to the data.
    rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
    print('rms')
    print(rms)

    # flag2 = 'rms %s  ' % str(rms)

    # plt.title('Accuracy')
    # plt.plot(fitted_model.history['acc'], label='train')
    # plt.plot(fitted_model.history['val_acc'], label='test')
    # plt.legend()
    # plt.show()

    # Visualising the results
    plt.plot(valid, color='red', label='Real Stock Price')
    plt.plot(closing_price, color='blue', label='Predicted Stock Price')
    plt.title('Stock Price Prediction  ')
    plt.xlabel('priceStringDate')
    plt.ylabel('price')
    plt.legend()
    plt.savefig('%s.png' % flag, bbox_inches='tight')
    plt.show()

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(valid, color='blue', label='Actual EVOO Stock Price')
    plt.plot(closing_price, color='red', label='Predicted EVOO Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('priceStringDate')
    plt.ylabel('price')
    plt.legend()

    plt.show()

    count = count + 1
