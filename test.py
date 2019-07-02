import requests
import json
import pandas as pd
import numpy as np
import scipy
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


#take json data from API
response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
print(json.dumps(parsed, indent=4))

#dataframe
df=pd.DataFrame(parsed)
#print(df)

 #separate data based on dataSource
dfOKAA = df[df['dataSource'] == 'OKAA']
print (dfOKAA)
dfEurop = df[df['dataSource'] == 'European Commission']
#print(dfEurop)
dfFAO = df[df['dataSource'] == 'FAO']
#print(dfFAO)

#Counting the number of unique countries in the dataset.

nunFAO=dfFAO['country'].nunique()
print(nunFAO)
nunEurop=dfFAO['country'].nunique()
print(nunEurop)
nunOKAA=dfFAO['country'].nunique()
print(nunOKAA)

#Visualization
imageDir = "plots/"
df_milk = dfFAO[dfFAO['product'] == 'milk and milk products']
ax = plt.gca()
df_milk.plot(kind='line',x='priceStringDate',y='price',ax=ax)
plt.savefig(imageDir + 'milk.png', bbox_inches='tight')
plt.show()

#hist
hist1=dfOKAA['price'].hist()
print(hist1)
hist2=dfOKAA['price'].plot(kind='hist', bins=20)
print(hist2)

#line plots
line_plot=dfOKAA.plot.line(x='country', y='price', figsize=(8,6))
print(line_plot)

#box Plot
box_plot=dfOKAA.plot.box(figsize=(10,8))
print(box_plot)

#Hexagonal Plots
hexbin=dfOKAA.plot.hexbin(x='price', y='price', gridsize=30, figsize=(8,6))
print(hexbin)

#Kernel Density Plots
kern=dfOKAA['price'].plot.kde()
print(kern)


#predictions
#Moving Average
#setting index as date
df['priceStringDate'] = pd.to_datetime(df.priceStringDate,format='%Y-%m-%d')

df.index = df['priceStringDate']


#plot
plt.figure(figsize=(16,8))
plt.plot(df['price'], label='Close Price history')

#creating dataframe with date and the target variable
Data = df.sort_index(ascending=True, axis=0)
new_Data = pd.DataFrame(index=range(0,len(df)),columns=['priceStringDate', 'price'])

for i in range(0,len(Data)):
     new_Data['priceStringDate'][i] = Data['priceStringDate'][i]
     new_Data['price'][i] = Data['price'][i]


# splitting into train and validation
train = new_Data[:2987]
valid = new_Data[2987:]
new_Data.shape, train.shape, valid.shape
((1235, 2), (987, 2), (248, 2))
train['priceStringDate'].min(), train['priceStringDate'].max(), valid['priceStringDate'].min(), valid['priceStringDate'].max()

(Timestamp('2013-10-08 00:00:00'),
Timestamp('2017-10-06 00:00:00'),
Timestamp('2017-10-09 00:00:00'),
Timestamp('2018-10-08 00:00:00'))
# make predictions
preds = []
for i in range(0, 248):
     a = train['price'][len(train) - 248 + i:].sum() + sum(preds)
     b = a / 248
     preds.append(b)

# calculate rmse
rms = np.sqrt(np.mean(np.power((np.array(valid['price']) - preds), 2)))
print(rms)

# plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.plot(train['price'])
plt.plot(valid[['price', 'Predictions']])

#Linear Regression
data1 = df.sort_index(ascending=True, axis=0)

#creating a separate dataset
new_data = pd.DataFrame(index=range(0,len(df)),columns=['priceStringDate', 'Close'])

for i in range(0,len(data)):
    new_data['priceStringDate'][i] = data1['priceStringDate'][i]
    new_data['Price'][i] = data1['Price'][i]
#create features
from fastai.structured import  add_datepart
add_datepart(new_data, 'Date')
new_data.drop('Elapsed', axis=1, inplace=True)  #elapsed will be the time stamp

new_data['mon_fri'] = 0
for i in range(0,len(new_data)):
    if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0

#split into train and validation
train = new_data[:987]
valid = new_data[987:]

x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']

#implement linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#make predictions and find the rmse
preds = model.predict(x_valid)
rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
print(rms)

#plot
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index

plt.plot(train['Price'])
plt.plot(valid[['Price', 'Predictions']])
