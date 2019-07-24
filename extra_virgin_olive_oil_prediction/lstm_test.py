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


# Downloading the Data
df = pd.read_csv("food_dataset.csv")

dfevoo = df[df['product'] == 'μπανάνες']
dfevoo = dfevoo[dfevoo['country'] == 'greece']
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
print(dfevoo['price'])

#Data Visualization
plt.figure(figsize = (18,9))
plt.plot(dfevoo['price'])
plt.xlabel('priceStringDate',fontsize=18)
plt.ylabel('price',fontsize=18)
plt.show()

# dfevoo = dfevoo.drop(columns=['priceStringDate'])
# Splitting Data into a Training set and a Test set
dfevoo.index=dfevoo.priceStringDate
dfevoo.drop('priceStringDate', axis=1, inplace=True)

dataset=dfevoo.values
train_perc = 0.7
quit(0)
# Splitting Data into a Training set and a Test set

train_data = dataset[0:int(len(dataset) * train_perc),:]  # .sample(frac=0.8, random_state=0)
test_data = dataset[int(len(dataset) * train_perc):]
print(train_data)
print(test_data)

# train_data =np.array(train_data)
# test_data =np.array(test_data)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

look_back=1
x_train, y_train = [], []
for i in range(look_back,len(train_data)):
    x_train.append(scaled_data[i-look_back:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
# Dropout layer is added to avoid over-fitting, which is a phenomenon where a machine learning model performs better on the training data compared to the test data
model.add(Dropout(0.2))

#add three more LSTM and dropout layers to our model
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=100))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(loss='mean_squared_error',
                    optimizer='nadam',
                    metrics=['mean_absolute_error', 'mean_squared_error'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(x_train, y_train, epochs=500,
                    validation_split=0.2, verbose=1, callbacks=[early_stop])


inputs = dfevoo[len(dfevoo) - len(test_data) - look_back:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(look_back,inputs.shape[0]):
    X_test.append(inputs[i-look_back:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


#for plotting
plt.figure(figsize = (18,9))
plt.plot(closing_price,color='b', label='Predection')
plt.plot(test_data,color='orange', label='Real price')

plt.show()

#for plotting
train = dfevoo[0:int(len(dfevoo) * train_perc)]  # .sample(frac=0.8, random_state=0)
valid = dfevoo[int(len(dfevoo) * train_perc):]
valid['Predictions'] = closing_price
plt.plot(train['price'])
plt.plot(valid[['price','Predictions']])
plt.show()
plt.savefig('%s.png' % flag, bbox_inches='tight')
quit(0)














