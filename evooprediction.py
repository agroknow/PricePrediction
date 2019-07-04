import requests
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import LSTM
from keras.layers import Dense

# take json data from API
response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
#print(json.dumps(parsed, indent=4))

# dataframe
df = pd.DataFrame(parsed)
#print(df)
dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
#print(dfevoo)
ax = plt.gca()
dfevoo.plot(kind='line',x='priceStringDate',y='price',ax=ax, figsize=(18, 16))

# make predictions
#moving average
Data=dfevoo.drop(columns=['price_id', 'product','dataSource','priceDate','url','country']).sort_index(ascending=True, axis=0)

train = Data[:7052] #80%
valid = Data[7052:] #20%

print(Data.shape,train.shape, valid.shape)

trainmin = train['priceStringDate'].min()
trainmax = train['priceStringDate'].max()
validmin = valid['priceStringDate'].min()
validmax = valid['priceStringDate'].max()
print(trainmin, trainmax, validmin, validmax)


preds = []
for i in range(0, 1763):
    a = train['price'][len(train) - 1763 + i:].sum() + sum(preds)
    b = a / 1763
    preds.append(b)

# calculate rmse
rms = np.sqrt(np.mean(np.power((np.array(valid['price']) - preds), 2)))
print(rms)


#Long Short Term Memory (LSTM)

#setting index
Data.index = Data.priceStringDate
Data.drop('priceStringDate', axis=1, inplace=True)
#
# creating train and test sets
dataset = Data.values
#
train = dataset[0:7052,:]
valid = dataset[7052:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
#
x_train = []
y_train = []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)
#
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
#
# # create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))
#
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

 #predicting 246 values, using past 60 from the train data
inputs = Data[len(Data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
#
X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
#
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print(rms)

#for plotting
# train = new_Data[:55382]
# valid = new_Data[55382:]
# valid['Predictions'] = closing_price
# plt.plot(train['price'])
# plt.plot(valid[['price','Predictions']])