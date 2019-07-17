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

# from fastai.structured import add_datepart
scaler = MinMaxScaler(feature_range=(0, 1))

sns.set_style('darkgrid')

# take json data from API
response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
# print(json.dumps(parsed, indent=4))

# dataframe
df = pd.DataFrame(parsed)

print(df['dataSource'])
df_okaa = df[df['dataSource'] == 'OKAA']
df_okaa = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']

dataset = df_okaa
#Data cleaning
dataset.isna().any()

training_set=dataset['price']
training_set=pd.DataFrame(training_set)

# Feature Scaling Normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train=np.array(X_train)
y_train = np.array(y_train)
print(X_train)
print('x_train')

print(y_train)
print('y_train')
quit(0)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(y_train)
quit(0)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 5, batch_size = 32)

#evaluate the model
train_acc = regressor.evaluate(X_train, y_train, verbose=1)
print('train_acc')
print(train_acc)

quit(0)


















dataset = pd.read_csv('Google_Stock_Price_Train.csv', index_col="Date", parse_dates=True)

####### extra virgin olive oil (up to 0,8°)

dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
# print(dfevoo)
ax = plt.gca()
dfevoo.plot(kind='line', x='priceStringDate', y='price', ax=ax, figsize=(18, 16))
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
Data = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_index(
    by='priceStringDate')
print(dfevoo.info())
print(Data)

# Long Short Term Memory (LSTM)

# setting index
# Data.index = Data.priceStringDate
# Data.drop('priceStringDate', axis=1, inplace=True)

# creating train and test sets
dataset = Data.values

# print(dataset.info())

train = dataset[0:7052, :]
valid = dataset[7052:, :]

print(Data.shape, train.shape, valid.shape)

# Data Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Convert Training Data to Right Shape

x_train = []
y_train = []
# execute a loop that starts from 61st record and stores all the previous 60 records to the x_train list. The 61st record is stored in the y_trainlabels list.
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
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
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

# add a dense layer at the end of the model. The number of neurons in the dense layer will be set to 1 since we want to predict a single value in the output.
model.add(Dense(1))

# Model Compilation
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Algorithm Training
model.fit(x_train, y_train, epochs=10, batch_size=132, verbose=1)

# evaluate the model
train_acc = model.evaluate(x_train, y_train, verbose=1)
print('train_acc')
print(train_acc)

# Prepare our test inputs
inputs = Data[len(Data) - len(valid) - 60:].values

# Scale our test data
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

# final test input
X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])

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

# plotting
plt.figure(figsize=(10, 6))
plt.plot(Data, color='blue', label='Actual EVOO Stock Price')
plt.plot(closing_price, color='red', label='Predicted EVOO Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
