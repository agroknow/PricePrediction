import requests
import json
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score

# from fastai.structured import add_datepart
scaler = MinMaxScaler(feature_range=(0, 1))

sns.set_style('darkgrid')

# # take json data from API
# response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
# data = response.text
# parsed = json.loads(data)
# # print(json.dumps(parsed, indent=4))
#
# # dataframe
# df = pd.DataFrame(parsed)

df = pd.read_csv("food_dataset.csv")
# Preview the first 5 lines of the loaded data


# prouped_ordered = df.groupby(['product']).size().reset_index(name='counts').sort_values('counts')
# print(prouped_ordered)
####### extra virgin olive oil (up to 0,8°)

dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
# dfevoo = df[df['product'] == 'virgin olive oil (up to 2°)']


dfevoo = dfevoo[dfevoo['country'] == 'greece']

dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])

# Select all duplicate rows based on one column
# dfevoo[dfevoo.duplicated(['priceStringDate'])]
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
Data = dfevoo
# dfevoo = dfevoo.drop_duplicates(subset ="priceStringDate", keep = 'first')
# print(dfevoo)
print(df['country'])



# quit(0)
# Long Short Term Memory (LSTM)

# setting index
Data.index = Data.priceStringDate
Data.drop('priceStringDate', axis=1, inplace=True)

# creating train and test sets
dataset = Data.values
print(dataset)
# quit(0)
# print(dataset.info())

train = dataset[0:int(0.8 * (len(dataset))), :]
valid = dataset[int(0.2 * (len(dataset))):, :]

print(Data.shape, train.shape, valid.shape)

# Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
N = 70
imageDir = "plots/"
count = 53
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
    epo = 100
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
