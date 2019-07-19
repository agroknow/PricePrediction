#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
import matplotlib.pyplot as plt

# Downloading the Data
df = pd.read_csv("food_dataset.csv")

dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
dfevoo = dfevoo[dfevoo['country'] == 'greece']
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()

train_perc = 0.7

train = dfevoo[:int(len(dfevoo) * train_perc)]  # .sample(frac=0.8, random_state=0)
test= dfevoo.drop(train.index)


x_train=train.drop(columns=['price'])
y_train=train.drop(columns=['priceStringDate'])

x_test=test.drop(columns=['price'])
y_test=test.drop(columns=['priceStringDate'])


scaler = MinMaxScaler(feature_range=(0, 1))

x_train_scaled = scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test_scaled)

#using gridsearch to find the best parameter
params = {'n_neighbors':range(1, 20), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
knn = neighbors.KNeighborsRegressor()
model = GridSearchCV(knn, params, cv=5)

#fit the model and make predictions
model.fit(x_train,y_train)
preds = model.predict(x_test)

print(model.best_params_)
print(model.best_score_)

test['Predictions'] = 0
test['Predictions'] = preds
plt.plot(test[['price', 'Predictions']])
plt.plot(train['price'])
plt.show()

plt.figure(figsize = (18,9))
plt.plot(test['Predictions'])
plt.plot(test['price'])

plt.xlabel('priceStringDate',fontsize=18)
plt.ylabel('price',fontsize=18)
plt.show()