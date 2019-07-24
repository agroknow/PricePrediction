import random

import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# https://towardsdatascience.com/a-quick-start-of-time-series-forecasting-with-a-practical-example-using-fb-prophet-31c4447a2274

periods = 30

# for periods in range(20, 65, 5):

columns = ['ds', 'y']
df = pd.read_csv("food_dataset.csv")

# df = df[df['product'].str.contains('olive oil')]
df = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
# df = df[df['country'] == 'greece']
psd = pd.to_datetime(df['priceStringDate'])
df['ds'] = psd
df['y'] = df['price']
df = df.groupby('ds').mean().reset_index()
df = df.sort_values(by='ds')
df = df[columns]
df = df.set_index('ds').resample('d').ffill().reset_index()

random_vals = []
for i in range(0, len(df['y'])):
    random_vals.append(random.uniform(0.995, 1.005))
df['y'] = df['y'] * random_vals

train = df[:len(df) - periods]
test = df[len(df) - periods:]

for seasonality in range(7, 366, 5):
    m = Prophet(weekly_seasonality=False, seasonality_mode='multiplicative')
    m.add_seasonality(name='custom', period=seasonality, fourier_order=10)
    # m = Prophet()
    m.fit(train)

    future = m.make_future_dataframe(periods=periods)

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    plotted = forecast[len(forecast) - periods:]

    # plt.plot(forecast['yhat'].values)
    # plt.plot(forecast['trend'].values)
    # plt.plot(df['y'].values)
    # plt.title('price prediction' + str(periods))
    # plt.legend(['predicted', 'trend', 'actual'], loc='upper left')
    # plt.show()

    plt.plot(plotted['yhat_lower'].values)
    plt.plot(plotted['yhat'].values)
    plt.plot(plotted['yhat_upper'].values)
    # plt.plot(plotted['trend'].values)
    plt.plot(test['y'].values)
    plt.title('Olive Oil Price Prediction, seasonality: ' + str(seasonality))
    # plt.legend(['predicted_lower', 'predicted_upper', 'predicted_mean', 'trend', 'actual'], loc='upper left')
    plt.legend(['predicted_lower', 'predicted', 'predicted_upper', 'actual'], loc=(1.04,0))
    plt.savefig('plots/s_' + str(seasonality) + 'prophet_' + str(periods) + '.png')
    plt.show()

# fig1 = m.plot(forecast)
# plt.show()
