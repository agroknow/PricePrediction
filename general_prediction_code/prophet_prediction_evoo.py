#Prophet
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fbprophet import Prophet

#creating dataframe

response = requests.get("http://148.251.22.254:8080/price-api-1.0/price/findAll")
data = response.text
parsed = json.loads(data)
# print(json.dumps(parsed, indent=4))

# dataframe
df = pd.DataFrame(parsed)

df = df[df['product'] == 'extra virgin olive oil (up to 0,8°)']
df = df[df['country'] == 'greece']



df['priceStringDate'] = pd.to_datetime(df['priceStringDate'])


df=df.groupby('priceStringDate').mean().reset_index()

data=df.drop(columns=['price_id', 'product', 'priceDate', 'url', 'country', 'dataSource']).sort_values(
  by='priceStringDate')
new_data=data

print(data)
print(new_data)

new_data['priceStringDate'] = pd.to_datetime(new_data.Date,format='%Y-%m-%d')


#preparing data
new_data.rename(columns={'price': 'y', 'priceStringDate': 'ds'}, inplace=True)

#train and validation
train = new_data[0:int(0.8*(len(new_data))), :]
valid = new_data[int(0.8*(len(new_data))):, :]

#fit the model
model = Prophet()
model.fit(train)

#predictions
close_prices = model.make_future_dataframe(periods=len(valid))
forecast = model.predict(close_prices)

#rmse
forecast_valid = forecast['yhat'][55382:]
rms=np.sqrt(np.mean(np.power((np.array(valid['y'])-np.array(forecast_valid)),2)))
print('rms')
print(rms)
#plot
valid['Predictions'] = 0
valid['Predictions'] = forecast_valid.values

plt.plot(train['y'])
plt.plot(valid[['y', 'Predictions']])
plt.show()