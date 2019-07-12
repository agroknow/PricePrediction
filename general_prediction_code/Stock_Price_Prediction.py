import requests
import json
import pandas as pd
import numpy as np
# from pmdarima.arima import auto_arima
# #from fbprophet import Prophet
# from statsmodels.tsa.arima_model import ARMA
# from pandas.plotting import register_matplotlib_converters
# import scipy
# from sklearn import neighbors
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
#from fastai.structured import add_datepart
scaler = MinMaxScaler(feature_range=(0, 1))

sns.set_style('darkgrid')

# take json data from API
response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
#print(json.dumps(parsed, indent=4))

# dataframe
df = pd.DataFrame(parsed)
# print(df)

# separate data based on dataSource
dfOKAA = df[df['dataSource'] == 'OKAA']
#print(dfOKAA)
dfEurop = df[df['dataSource'] == 'European Commission']
# print(dfEurop)
dfFAO = df[df['dataSource'] == 'FAO']
# print(dfFAO)
prouped_ordered = df.groupby(['product']).size().reset_index(name='counts').sort_values('counts')

# Counting the number of unique countries in the dataset.

nunFAO = dfFAO['country'].nunique()
#print(nunFAO)
nunEurop = dfFAO['country'].nunique()
#print(nunEurop)
nunOKAA = dfFAO['country'].nunique()
#print(nunOKAA)

# #Visualization
imageDir = "plots/"
df_milk = dfFAO[dfFAO['product'] == 'milk and milk products']
ax = plt.gca()
df_milk.plot(kind='line',x='priceStringDate',y='price',ax=ax)
plt.savefig(imageDir + 'milk.png', bbox_inches='tight')
#plt.show()
# #Visualization
#
# # hist
#
# hist2 = dfOKAA['price'].plot(kind='hist', bins=100)
# print(hist2)
# plt.show()
#
# # create figure and axis
# fig, ax = plt.subplots()
# # plot histogram
# ax.hist(dfFAO['price'])
# # set title and labels
# ax.set_title('hist')
# ax.set_xlabel('price')
# ax.set_ylabel('den exw idea')
# plt.show()
# # line plots
# line_plot = dfOKAA.plot.line(x='priceStringDate', y='price', figsize=(8, 6))
# print(line_plot)
# plt.show()
# # box Plot
# box_plot = dfOKAA.plot.box(figsize=(10, 8))
# print(box_plot)
# plt.show()
#
# # Hexagonal Plots
# hexbin = dfOKAA.plot.hexbin(x='price', y='price', gridsize=30, figsize=(8, 6))
# print(hexbin)
# plt.show()
#
# # Kernel Density Plots
# kern = dfOKAA['price'].plot.kde()
# print(kern)
# plt.show()
#
#
# #scatter plot
#
# # create a figure and axis
# fig, ax = plt.subplots()
# # scatter the sepal_length against the sepal_width
# ax.scatter(dfFAO['priceStringDate'], dfFAO['price'])
# # set a title and labels
# ax.set_title('FAO Dataset')
# ax.set_xlabel('Date')
# ax.set_ylabel('price')
# plt.show()

# predictions

# #plot
# df_country = df[df['country'] == 'greece']
# #df_milk = dfFAO[dfFAO['product'] == 'milk and milk products']
#
# ax = plt.gca()
# df_country.plot(kind='line', x='priceStringDate', y='price', ax=ax)
# #plt.show()
#
# # Moving Average
#
# # creating dataframe with date and the target variable
Data = df.sort_index(ascending=True, axis=0) #sorting
new_Data = pd.DataFrame(index=range(0, len(df)), columns=['priceStringDate', 'price']) #create a separate dataset

for i in range(0, len(Data)):
    new_Data['priceStringDate'][i] = Data['priceStringDate'][i]
    new_Data['price'][i] = Data['price'][i]



# splitting into train and validation
train = new_Data[:55382] #80%
valid = new_Data[55382:] #20%
#
print(new_Data.shape,train.shape, valid.shape)

trainmin = train['priceStringDate'].min()
trainmax = train['priceStringDate'].max()
validmin = valid['priceStringDate'].min()
validmax = valid['priceStringDate'].max()
print(trainmin, trainmax, validmin, validmax)

# make predictions
preds = []
for i in range(0, 13846):
    a = train['price'][len(train) - 13846 + i:].sum() + sum(preds)
    b = a / 13846
    preds.append(b)

# calculate rmse
rms = np.sqrt(np.mean(np.power((np.array(valid['price']) - preds), 2)))
print(rms)

#MA with ARMA class
#code

# fit model
# print(np.asarray(new_Data))
# model = ARMA(new_Data, order=(0, 1)) #first-order moving average model.
# model_fit = model.fit(disp=False)
# # make prediction
# yhat = model_fit.predict(len(new_Data), len(new_Data))
# print(yhat)

# plot de paizei swsta
# valid['Predictions'] = 0
# valid['Predictions'] = preds
# plt.plot(train['price'])
# plt.show()
# plt.plot(valid[['price', 'Predictions']])
# plt.show()
#


# Linear Regression

#create features
#add_datepart(new_Data, 'priceStringDate')
# new_Data.drop('Elapsed', axis=1, inplace=True)  # elapsed will be the time stamp
#
# #
# x_train = train.drop('price', axis=1)
# y_train = train['price']
# x_valid = valid.drop('price', axis=1)
# y_valid = valid['price']
# #
# # implement linear regression
# model = LinearRegression()
# model.fit(x_train, y_train)
# #
# # make predictions and find the rmse
# preds = model.predict(x_valid)
# rms = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(preds)), 2)))
# print(rms)
#
# # plot
# # valid['Predictions'] = 0
# # valid['Predictions'] = preds
# #
# # valid.index = new_Data[55382:].index
# # train.index = new_Data[:55382].index
# #
# # plt.plot(train['price'])
# # plt.plot(valid[['price', 'Predictions']])
#
# #k- Nearest Neighbours
#
# #scaling data
# x_train_scaled = scaler.fit_transform(x_train)
# x_train = pd.DataFrame(x_train_scaled)
# x_valid_scaled = scaler.fit_transform(x_valid)
# x_valid = pd.DataFrame(x_valid_scaled)
#
# #using gridsearch to find the best parameter
# params = {'n_neighbors':[2,3,4,5,6,7,8,9]}
# knn = neighbors.KNeighborsRegressor()
# model = GridSearchCV(knn, params, cv=5)
#
# #fit the model and make predictions
# model.fit(x_train,y_train)
# preds = model.predict(x_valid)
#
# #rmse
# rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
# print('rms from knn')
# print(rms)
# #plot
# # valid['Predictions'] = 0
# # valid['Predictions'] = preds
# # plt.plot(valid[['price', 'Predictions']])
# # plt.plot(train['price'])
#

# Auto Arima
# Data = df.sort_index(ascending=True, axis=0)
# #Data = dfFAO.sort_index(ascending=True, axis=0)
# train = Data[:55382]
# valid = Data[55382:]
# #train = Data[:3000]
# #valid = Data[3000:]
# training = train['price']
# validation = valid['price']
#
# model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12,start_P=0, seasonal=True, d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
# model.fit(training)
#
# forecast = model.predict(n_periods=13846)
# #forecast = model.predict(n_periods=133)
# forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])
#
# rms=np.sqrt(np.mean(np.power((np.array(valid['price'])-np.array(forecast['Prediction'])),2)))
# print('auto arima')
# print(rms)
# #plot
# plt.plot(train['price'])
# plt.plot(valid['price'])
# plt.plot(forecast['Prediction'])

#Prophet
#creating dataframe
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['priceStringDate', 'price'])
#
# for i in range(0,len(data)):
#     new_data['priceStringDate'][i] = data['priceStringDate'][i]
#     new_data['price'][i] = data['price'][i]
#
# new_data['priceStringDate'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')
# new_data.index = new_data['priceStringDate']
#
# #preparing data
# new_data.rename(columns={'price': 'y', 'priceStringDate': 'ds'}, inplace=True)
#
# #train and validation
# train = new_data[:55382]
# valid = new_data[55382:]
#
# #fit the model
# model = Prophet()
# model.fit(train)
#
# #predictions
# close_prices = model.make_future_dataframe(periods=len(valid))
# forecast = model.predict(close_prices)
#
# #rmse
# forecast_valid = forecast['yhat'][55382:]
# rms=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
# print(rms)
#plot
# valid['Predictions'] = 0
# valid['Predictions'] = forecast_valid.values
#
# plt.plot(train['y'])
# plt.plot(valid[['y', 'Predictions']])


#Long Short Term Memory (LSTM)

#setting index
new_Data.index = new_Data.priceStringDate
new_Data.drop('priceStringDate', axis=1, inplace=True)


#creating train and test sets
dataset = new_Data.values
print(dataset)
train = dataset[0:55382,:]
valid = dataset[55382:,:]
print(train)
#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train = []
y_train = []
for i in range(1111,len(train)):
    x_train.append(scaled_data[i-1111:i,0])
    y_train.append(scaled_data[i,0])
x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)

#predicting 246 values, using past 60 from the train data
inputs = new_Data[len(new_Data) - len(valid) - 1111:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(1111,inputs.shape[0]):
    X_test.append(inputs[i-1111:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print(rms)

#for plotting
train = new_Data[:55382]
valid = new_Data[55382:]
valid['Predictions'] = closing_price
plt.plot(train['price'])
plt.plot(valid[['price','Predictions']])
plt.show()