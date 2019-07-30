# importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima

df = pd.read_csv("food_dataset.csv")

dfevoo = df[df['product'].str.contains('χοιρ')]
dfevoo = dfevoo[dfevoo['country'] == 'greece']
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
data = dfevoo['price']
train_perc = 0.7

def difference(dataset, interval=1):
    index = list(dataset.index)
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset["price"][i] - dataset["price"][i - interval]
        diff.append(value)
    return (diff)

training = difference(dfevoo)
train = dfevoo[:int(len(dfevoo) * train_perc)]  # .sample(frac=0.8, random_state=0)
test = dfevoo.drop(train.index)
#


# training = train['price']
# testing = test['price']

plt.figure(figsize=(18, 9))
plt.plot(dfevoo['price'])
plt.xlabel('priceStringDate', fontsize=18)
plt.ylabel('price', fontsize=18)
plt.show()


k = 1
for i in range(60, 80, 5):
    # model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
    model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=80, start_P=1, start_Q=1, max_P=3,
                       max_Q=3, stepwise=True, seasonal=True, d=1, D=1, trace=True)

    model.fit(training)

    forecast = model.predict(n_periods=len(test))
    forecast = pd.DataFrame(forecast, index=test.index, columns=['Prediction'])
    # plot
    plt.plot(train['price'])
    plt.plot(test['price'], color='orange', label='Real price')
    plt.plot(forecast['Prediction'], color='b', label='Prediction')
    plt.savefig('%s.png' % i, bbox_inches='tight')

    plt.show()

plt.plot(test['price'], color='orange', label='Real price')
plt.plot(forecast['Prediction'], color='b', label='Prediction')
plt.show()
