import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

test_date = '2019-01-01'
columns = ['ds', 'y']
df = pd.read_csv("food_dataset.csv")

df = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
df = df[df['country'] == 'greece']
psd = pd.to_datetime(df['priceStringDate'])
df['ds'] = psd
df['y'] = df['price']
df = df[df['ds'] < test_date]
df = df[columns]

print(df.head())

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=76)
print(future.tail())


df = pd.read_csv("food_dataset.csv")

df = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
df = df[df['country'] == 'greece']
psd = pd.to_datetime(df['priceStringDate'])
df['ds'] = psd
df['y'] = df['price']
df = df[df['ds'] >= test_date]
df = df[columns]

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

print(forecast.info())

plt.plot(forecast['yhat'].values)
plt.plot(df['y'].values)
# plt.plot(forecast['yhat_lower'].values)
# plt.plot(forecast['yhat_upper'].values)
plt.title('price prediction')
plt.legend(['predicted', 'actual'], loc='upper left')
# plt.legend(['predicted', 'min', 'max'], loc='upper left')
plt.show()

fig2 = m.plot_components(forecast)
plt.show()
print(fig2)


