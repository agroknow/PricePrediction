# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import json

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# print(check_output(["ls", "./input"]).decode("utf8"))
#
# # Any results you write to the current directory are saved as output.
#
# data = pd.read_csv('./input/all_stocks_5yr.csv')
# cl = data[data['Name'] == 'MMM'].Close

# FETCH ON A DAY BASIS
# product_query = "{ \"aggregations\": { \"years\": { \"attribute\": \"products.value.keyword\", \"size\": 15000 }}, \"apikey\": \"32aa618e-be8b-32da-bc99-5172f90a903e\", \"detail\": true, \"entityType\": \"incident\"}"
# headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
# apikey = "32aa618e-be8b-32da-bc99-5172f90a903e"
# data = requests.post("http://148.251.22.254:8080/search-api-1.0/search/", data=product_query, headers=headers)
#
# print(data.json())
#
# with open('products.json', 'w') as outfile:
#     json.dump(data.json(), outfile)
# data = json.loads(open('products.json', 'r').read())
#
# threshold = 50
# products = []
#
# for bucket in data['aggregations']['sterms#years']['buckets']:
#     if bucket['doc_count'] > threshold:
#         specific_product = "{ \"aggregations\": { \"days\": { \"attribute\": \"createdOn\", \"format\": \"yyyy-MM-dd\", \"interval\": \"DAY\", \"size\": 20000 }}, \"apikey\": \"32aa618e-be8b-32da-bc99-5172f90a903e\", \"detail\": true, \"entityType\": \"incident\",\"product\": \"" + \
#                            bucket['key'] + "\"}"
#         sp_data = requests.post("http://148.251.22.254:8080/search-api-1.0/search/", data=specific_product,
#                                 headers=headers)
#
#         sp_data = sp_data.json()
#
#         for sp_bucket in sp_data['aggregations']['date_histogram#days']['buckets']:
#             products.append([bucket['key'], sp_bucket['key_as_string'], sp_bucket['doc_count']])
#
#
# with open('dates_products.json', 'w') as outfile:
#     json.dump(products, outfile)
# products = json.loads(open('dates_products.json', 'r').read())
# ENDOF FETCH DAY BASIS



# FETCH ON A MONTH BASIS
product_query = "{ \"aggregations\": { \"years\": { \"attribute\": \"products.value.keyword\", \"size\": 500, \"subAggregation\": { \"attribute\": \"createdOn\", \"format\": \"YYYY-MM\", \"interval\": \"MONTH\", \"size\": 500 } }}, \"apikey\": \"32aa618e-be8b-32da-bc99-5172f90a903e\", \"detail\": true, \"entityType\": \"incident\"}"
headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
apikey = "32aa618e-be8b-32da-bc99-5172f90a903e"
data = requests.post("http://148.251.22.254:8080/search-api-1.0/search/", data=product_query, headers=headers)

print(data.json())

with open('sample.json', 'w') as outfile:
    json.dump(data.json(), outfile)

data = json.loads(open('sample.json', 'r').read())

threshold = 50
products = []
for bucket in data['aggregations']['sterms#years']['buckets']:
    if bucket['doc_count'] > threshold:

        for subagg in bucket['date_histogram#subaggregation']['buckets']:
            products.append([bucket['key'], subagg['key_as_string'], subagg['doc_count']])
print(products)
#END OF FETCH MONTH BASIS




df = pd.DataFrame(products, columns=['product', 'date', 'count'])
print(df.info())
print(df.head())

df['date'] = pd.to_datetime(df['date'])

df = df.sort_values(by='date')
to_use = df[df['product'] == 'nuts, nut products and seeds']
to_use = df[df['product'] == 'non-alcoholic beverages']
to_use = df[df['product'] == 'fruits and vegetables']
# print(to_use.head(150))
# quit(0)
cl = to_use['count']

scl = MinMaxScaler()
# Scale the data
cl = cl.values.reshape(cl.shape[0], 1)
cl = scl.fit_transform(cl)


# Create a function to process the data into day_lookup day look back slices
def processData(data, lb):
    X, Y = [], []
    for i in range(len(data) - lb - 1):
        X.append(data[i:(i + lb), 0])
        Y.append(data[(i + lb), 0])
    return np.array(X), np.array(Y)


day_lookup = 7

X, y = processData(cl, day_lookup)
X_train, X_test = X[:int(X.shape[0] * 0.80)], X[int(X.shape[0] * 0.80):]
y_train, y_test = y[:int(y.shape[0] * 0.80)], y[int(y.shape[0] * 0.80):]
print(X_train.shape[0])
print(X_test.shape[0])
print(y_train.shape[0])
print(y_test.shape[0])

# Build the model
model = Sequential()
model.add(LSTM(256, input_shape=(day_lookup, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# Reshape data for (Sample,Timestep,Features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
# Fit model with history to check for overfitting
history = model.fit(X_train, y_train, epochs=300, validation_data=(X_test, y_test), shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1, 1)))
plt.plot(scl.inverse_transform(Xt))

act = []
pred = []
# for i in range(250):
i = len(X_test) - 1
Xt = model.predict(X_test[i].reshape(1, day_lookup, 1))
print('predicted:{0}, actual:{1}'.format(scl.inverse_transform(Xt), scl.inverse_transform(y_test[i].reshape(-1, 1))))
pred.append(scl.inverse_transform(Xt))
act.append(scl.inverse_transform(y_test[i].reshape(-1, 1)))

result_df = pd.DataFrame({'pred': list(np.reshape(pred, (-1))), 'act': list(np.reshape(act, (-1)))})

print(X_test)

Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1, 1)))
plt.plot(scl.inverse_transform(Xt))
plt.show()
