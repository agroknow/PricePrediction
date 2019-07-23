import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("food_dataset.csv")

dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
dfevoo = dfevoo[dfevoo['country'] == 'greece']
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
data = dfevoo['price']
train_perc = 0.7

sns.distplot(dfevoo['price'])
# plt.show()
X = dfevoo[['priceStringDate']]  # .sample(frac=0.8, random_state=0)
y = dfevoo['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression(n_jobs=-1)
lm.fit(X_train,y_train)
X_test=X_test.values
X_test=X_test.astype(float)
predictions = lm.predict(X_test)
X=X.values
y=y.values
plt.scatter(y_test,predictions,c='black')
plt.show()
plt.scatter(X,y)

plt.show()
plt.figure(figsize = (18,9))
plt.plot(predictions,color='b', label='Predection')
plt.plot(y,color='orange', label='Real price')
plt.show()


