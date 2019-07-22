#importing libraries
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
df = pd.read_csv("food_dataset.csv")

dfevoo = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
dfevoo = dfevoo[dfevoo['country'] == 'greece']
dfevoo['priceStringDate'] = pd.to_datetime(dfevoo['priceStringDate'])
dfevoo = dfevoo.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
    by='priceStringDate')
dfevoo = pd.DataFrame(dfevoo)
dfevoo = dfevoo.groupby('priceStringDate').mean().reset_index()
data=dfevoo['price']
train_perc = 0.7

train = dfevoo[:int(len(dfevoo) * train_perc)]  # .sample(frac=0.8, random_state=0)
test= dfevoo.drop(train.index)

training=train['price']
testing=test['price']

# model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
# model = auto_arima(data, start_p=0, start_q=0,max_p=30, max_q=30, m=100,start_P=0, start_Q=0, max_P=30, max_Q=30, stepwise=True, seasonal=True,d=1,n_fits=2, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model = auto_arima(training, start_p=1, start_q=1,max_p=1, max_q=1, m=50,start_P=1, start_Q=1, max_P=1, max_Q=1, stepwise=True,random_state=5,n_fits=50, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)


model.fit(training)

forecast = model.predict(n_periods=len(test))
forecast = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
#plot
plt.plot(train['price'])
plt.plot(test['price'],color='orange', label='Real price')
plt.plot(forecast['Prediction'],color='b', label='Prediction')
plt.show()


plt.plot(test['price'],color='orange', label='Real price')
plt.plot(forecast['Prediction'],color='b', label='Prediction')
plt.show()

