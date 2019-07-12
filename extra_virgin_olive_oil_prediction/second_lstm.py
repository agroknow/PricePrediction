import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#from sklearn.cross_validation import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis

df = pd.read_csv('food_dataset.csv', header=0)
#print(df)
dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
#dfevoo = df[df['product'] == 'virgin olive oil (up to 2°)']



dfevoo = dfevoo[dfevoo['country'] == 'greece']


dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])

dfevoo=dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
  by='priceStringDate')
dfevoo=pd.DataFrame(dfevoo)

Data=dfevoo


dfevoo=dfevoo.groupby('priceStringDate').mean().reset_index()

# setting index
Data.index = Data.priceStringDate
Data.drop('priceStringDate', axis=1, inplace=True)

#creating train and test sets
dataset = Data.values
print(dataset)
plt.plot(dataset)
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train = dataset[0:int(0.8*(len(dataset))), :]
valid = dataset[int(0.8*(len(dataset))):, :]

print(Data.shape, train.shape, valid.shape)

# convert an array of values into a dataset matrix
def create_dataset(data, look_back):
	dataX, dataY = [], []
	for i in range(len(data)-look_back-1):
		a = data[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(data[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(valid, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(trainX.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(output_dim=1))


start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print ('compilation time : ', time.time() - start)

history=model.fit(trainX,trainY,batch_size=128,nb_epoch=10,validation_split=0.05)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()



# def plot_results_multiple(predicted_data, true_data, length):
#     plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1))[length:])
#     plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))[length:])
#     plt.show()
#
#
# # predict lenght consecutive values from a real one
# def predict_sequences_multiple(model, firstValue, length):
#     prediction_seqs = []
#     curr_frame = firstValue
#
#     for i in range(length):
#         predicted = []
#
#         print(model.predict(curr_frame[newaxis, :, :]))
#         predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
#
#         curr_frame = curr_frame[0:]
#         curr_frame = np.insert(curr_frame[0:], i + 1, predicted[-1], axis=0)
#
#         prediction_seqs.append(predicted[-1])
#
#     return prediction_seqs


predict_length = 5
predictions = predict_sequences_multiple(model, testX[0], predict_length)
print(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)))
plot_results_multiple(predictions, testY, predict_length)